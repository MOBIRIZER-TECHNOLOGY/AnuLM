"""
Sarvam-105B  ::  YaRN rotary construction  -- annotated, with the config's
numbers actually worked out.

Extracts (code verbatim, comments added) from sarvam-105b/modeling_sarvam_moe.py:
    111-145   SarvamMLARotaryEmbedding        (the plain RoPE base class)
    147-168   yarn_find_correction_dim / _range / get_mscale / linear_ramp_mask
    170-225   SarvamMLAYarnRotaryEmbedding    (the override that does the work)
    235-247   apply_rotary_pos_emb            (note the layout permutation)
Fourth companion to mla_/gqa_/moe_routing_annotated.py.

Config inputs (sarvam-105b/config.json), all substituted below:

    dim                              = qk_rope_head_dim  = 64   <- NOT head_dim
    base                             = rope_theta        = 10000
    factor (scaling_factor)          = 40
    original_max_position_embeddings = 4096
    beta_fast / beta_slow            = 32 / 1
    mscale / mscale_all_dim          = 1.0 / 1.0
    max_position_embeddings          = 131072   = 4096 x 32   (note: not x40)

The one-line summary: RoPE's frequency spectrum is split into three bands.
High-frequency dims keep their original frequencies, low-frequency dims are
divided by 40, and a linear ramp blends the thirteen dims in between.
"""


# =============================================================================
# PART 0 -- WHAT PLAIN ROPE DOES (the base class, line 111)
# =============================================================================

class SarvamMLARotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        # dim = 64: only the rope half of the 192-d query is rotated. The 128-d
        # nope half never touches this module. See mla_forward_annotated.py.
        self.dim = dim
        self.base = base
        # Classic geometric frequency ladder over dim//2 = 32 pairs:
        #   inv_freq[i] = base ** (-2i/dim),  i = 0..31
        # i=0 is the fastest (wavelength 6.3 tokens), i=31 the slowest (47117).
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)   # (32,)

        self._set_cos_sin_cache(seq_len=max_position_embeddings, ...)
        self.max_seq_len_cached = None
        # ^ Note the order: the cache is BUILT and then the length marker is
        #   reset to None, so the first forward() rebuilds it. With
        #   max_position_embeddings = 131072 that init build is
        #     131072 * 64 * 4 bytes * 2 (cos+sin) = 67.1 MB per layer
        #   and _init_rope gives every attention layer its own instance, so
        #   32 layers = 2.15 GB allocated and immediately orphaned. Transient,
        #   but it is a real spike if you are tight on memory at load time.

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, ...)             # (T,)
        freqs = torch.outer(t, self.inv_freq.to(t.device))         # (T, 32)
        # Duplicated, not interleaved -- pairs the halves for rotate_half().
        emb = torch.cat((freqs, freqs), dim=-1)                    # (T, 64)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # Grow-only cache: any request longer than what is cached rebuilds the
        # whole table. Returns (T, 64) -- no batch or head axis yet.
        if self.max_seq_len_cached is None or seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return self.cos_cached[:seq_len].to(x.dtype), self.sin_cached[:seq_len].to(x.dtype)


# =============================================================================
# PART 1 -- WHY NAIVE EXTENSION FAILS, AND THE TWO BAD FIXES
# =============================================================================
# Dimension i rotates at inv_freq[i] radians per token, i.e. wavelength
# lambda_i = 2*pi / inv_freq[i] tokens. Over the 4096-token training window it
# completes 4096/lambda_i rotations. Across the 32 pairs that ranges from ~652
# rotations (i=0) down to ~0.1 (i=31). So the ladder is not one mechanism but
# two:
#
#   FAST dims wrap around many times inside the training window. The model has
#   seen every phase of them, so they can only encode LOCAL relative offset --
#   "three tokens back". Position 5 and position 4101 look identical to them.
#
#   SLOW dims complete less than one rotation even at 4096. The model has never
#   seen their later phases. They encode ABSOLUTE position -- "roughly 60% of
#   the way through" -- and they are the dims that break when you go past 4096,
#   because you are asking for phase values that were never trained.
#
# The two naive fixes each sacrifice one band:
#
#   Position Interpolation: divide every position by 40. The slow dims are now
#   in-distribution -- but the fast dims get squashed 40x too, so neighbouring
#   tokens become nearly indistinguishable and local attention degrades.
#
#   Pure extrapolation (raise base, as the 30B does with theta=8e6): keeps the
#   fast dims intact, but only works if you actually TRAINED at the long length.
#   Bolted onto a 4096-trained model it produces out-of-distribution phases.
#
# YaRN's observation: apply each fix to the band it suits. Interpolate the slow
# dims, extrapolate the fast ones, ramp between. "NTK-by-parts."


# =============================================================================
# PART 2 -- WHERE TO PUT THE BOUNDARIES (lines 147-156)
# =============================================================================

def yarn_find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=2048):
    # Inverts "how many rotations does dim i complete over the training window"
    # to "which dim i completes exactly num_rotations". Solve for i in
    #     max_pos / lambda_i = num_rotations,  lambda_i = 2*pi*base**(2i/dim)
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def yarn_find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
    # beta_fast=32 -> the dim that turns 32 times over 4096 -> 10.4722 -> low=10
    # beta_slow=1  -> the dim that turns  1 time  over 4096 -> 22.5134 -> high=23
    low = math.floor(yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)
    # For this config: (10, 23).  The betas are thresholds in ROTATIONS, and
    # they are measured against original_max_position_embeddings = 4096 -- the
    # length the model was actually trained at, never the 131072 target.


def yarn_linear_ramp_mask(min_val, max_val, dim):
    # Straight line from 0 at i=low to 1 at i=high, clamped outside.
    if min_val == max_val:
        max_val += 0.001                  # guard against a divide-by-zero
    linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
    return torch.clamp(linear_func, 0, 1)                          # (32,)


# =============================================================================
# PART 3 -- BUILDING THE BLENDED SPECTRUM (line 192)
# =============================================================================

def _set_cos_sin_cache(self, seq_len, device, dtype):   # SarvamMLAYarnRotaryEmbedding
    self.max_seq_len_cached = seq_len
    dim = self.dim                                                 # 64

    # Two complete frequency ladders, differing only by the /40.
    freq_extra = 1.0 / (self.base ** (torch.arange(0, dim, 2, ...) / dim))
    #   unmodified: EXTRAPOLATION ladder                           (32,)
    freq_inter = 1.0 / (self.scaling_factor * self.base ** (torch.arange(0, dim, 2, ...) / dim))
    #   every frequency / 40: INTERPOLATION ladder                 (32,)
    # Note this divides the FREQUENCY, which is identical to dividing the
    # position index -- the same thing Position Interpolation does, just
    # expressed on the other side of the product.

    low, high = yarn_find_correction_range(
        self.beta_fast, self.beta_slow, dim, self.base,
        self.original_max_position_embeddings)                     # -> (10, 23)

    # ramp: 0 below dim 10, 1 above dim 23. mask = its complement, so
    # mask = 1 means "keep the original frequency".
    inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2).to(...)  # (32,)

    # The blend. Per dim i:
    #     mask=1 -> freq_extra   (fast dims, untouched)
    #     mask=0 -> freq_inter   (slow dims, /40)
    #     between -> linear mix
    inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
    self.register_buffer("inv_freq", inv_freq, persistent=False)   # (32,)
    #
    # Worked out for this config (wavelengths in tokens, rotations over 4096):
    #
    #     i   lambda_i    rot@4096   mask   what happens
    #     --  ----------  ---------  -----  ---------------------------
    #      0        6.3      651.9   1.000  extrapolate, unchanged
    #      5       26.5      154.6   1.000  extrapolate, unchanged
    #     10      111.7       36.7   1.000  extrapolate, unchanged  <- last
    #     11      149.0       27.5   0.923  blend begins
    #     15      471.2        8.7   0.615  blend
    #     18     1117.3        3.7   0.385  blend
    #     22     3533.3        1.2   0.077  blend ends
    #     23     4711.7        0.9   0.000  interpolate, /40        <- first
    #     31    47117.2        0.1   0.000  interpolate, /40
    #
    # Sanity check on the betas: dim 10 turns 36.7 times over the training
    # window (just past beta_fast=32) and dim 23 turns 0.9 times (just under
    # beta_slow=1). The cutoffs land exactly where they were asked to.
    # 11 of 32 dims are untouched, 13 blend, 8 are fully interpolated.

    t = torch.arange(seq_len, device=device, dtype=torch.float32)  # (T,)
    freqs = torch.outer(t, inv_freq)                               # (T, 32)

    # ---- the temperature term ------------------------------------------
    # yarn_get_mscale(scale, m) = 0.1*m*ln(scale) + 1.  Here:
    #     numerator   = get(40, mscale=1.0)         = 1.36889
    #     denominator = get(40, mscale_all_dim=1.0) = 1.36889
    #     _mscale     = 1.00000  exactly
    # So in THIS config the rotary tables are NOT scaled at all. The ratio form
    # exists so the correction can be split between the embeddings and the
    # attention logits; Sarvam puts all of it in the logits instead, back in
    # SarvamMLAAttention.__init__:522:
    #     softmax_scale = 192**-0.5 * 1.36889**2 = 0.07217 * 1.87385 = 0.13523
    # Same 1.874x factor either way. The point of it: stretching context 40x
    # spreads attention over more keys and raises softmax entropy, so the
    # logits get a compensating gain. That gain is the third leg of YaRN,
    # alongside the blend -- easy to miss because it lives in a different class.
    _mscale = float(
        yarn_get_mscale(self.scaling_factor, self.mscale)
        / yarn_get_mscale(self.scaling_factor, self.mscale_all_dim)
    )

    emb = torch.cat((freqs, freqs), dim=-1)                        # (T, 64)
    self.register_buffer("cos_cached", (emb.cos() * _mscale).to(dtype), persistent=False)
    self.register_buffer("sin_cached", (emb.sin() * _mscale).to(dtype), persistent=False)
    #   cos_cached / sin_cached                                    (T, 64)


# =============================================================================
# PART 4 -- APPLYING IT, AND THE LAYOUT PERMUTATION (line 235)
# =============================================================================

def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    # Table lookup by position, then a head axis of size 1 to broadcast over
    # all 64 heads.  cos/sin: (T, 64) -> (B, S, 64) -> (B, 1, S, 64)
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)

    # ---- THE TRAP -------------------------------------------------------
    # This de-interleaves before rotating, and the 30B's version does not.
    #   (b,h,s,64) -> view (b,h,s,32,2) -> transpose(4,3) -> (b,h,s,2,32)
    #               -> reshape (b,h,s,64)
    # i.e. [x0,x1,x2,x3,...] becomes [x0,x2,x4,...,x1,x3,x5,...].
    # So the 105B checkpoint stores rope dims INTERLEAVED (GPT-J convention)
    # while rotate_half below expects the SPLIT-HALVES layout (GPT-NeoX). This
    # permutation converts between them. Port the weights without it and you
    # get a model that looks fine on short prompts and degrades with distance.
    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    # The permutation is never undone -- but it is applied identically to q_pe
    # and k_pe, and <Pq, Pk> = <q, k> for any shared permutation P, so the
    # attention logits are unaffected. Only the two rope halves pass through
    # here; q_nope/k_nope and v are untouched.

    q_embed = (q * cos) + (rotate_half(q) * sin)                   # (B, 64, S, 64)
    k_embed = (k * cos) + (rotate_half(k) * sin)                   # (B,  1, S, 64)
    return q_embed, k_embed


# =============================================================================
# CLOSING NOTES
# =============================================================================
#
# 1. THE CORRECTION IS SPLIT ACROSS TWO CLASSES. The frequency blend lives in
#    the rotary module; the 1.874x temperature gain lives in the attention
#    module's __init__. With mscale == mscale_all_dim the rotary's own _mscale
#    collapses to exactly 1.0, so reading only this file you would conclude
#    YaRN's temperature term was unused. It is not -- it moved.
#
# 2. THE BETAS ARE MEASURED AGAINST 4096, NOT 131072. Every threshold in
#    yarn_find_correction_range uses original_max_position_embeddings. That
#    field is the length the model was pretrained at, and getting it wrong
#    silently moves the band boundaries and changes which dims are protected.
#    It is the field most often mis-set when people copy a rope_scaling block
#    between models.
#
# 3. 40x SCALING FOR A 32x EXTENSION. factor=40 but 131072/4096 = 32. The
#    factor is deliberately generous relative to the actual extension -- a
#    margin so the interpolated band still has headroom at full length. The
#    code never checks the two against each other; factor drives the frequency
#    division and the mscale, max_position_embeddings only sizes the table.
#
# 4. CONTRAST WITH THE 30B, which solves the same problem by training at length
#    with rope_theta = 8e6 and rope_scaling = null -- no bands, no ramp, no
#    temperature correction, flat 64**-0.5 scale. Raising the base is pure
#    extrapolation and only works because they trained that way; YaRN is what
#    you reach for when the long context is bolted on afterwards. The two
#    Sarvam models are a clean side-by-side of both strategies.
