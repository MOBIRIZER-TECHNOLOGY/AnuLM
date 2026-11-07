# Launch checklist: the public page and the hosted demo

What is built, what each remaining step needs, and who decides it. The research
behind the choices (hosting costs, India's DPDP Act, licensing wording, comparable
projects) is in the report *AnuLM public launch landing page* (2026-11-07).

## Built and tested

| piece | where | tested |
|---|---|---|
| Landing page with "Work with us": three lanes (colleges and students, companies, researchers), an enquiry form, the licence in plain words | `site/index.html`, `site/styles.css`, `site/main.js` | desktop and 360 px phone width, light and dark; every form path (empty, under 18, spam trap, sent, failed, not connected) |
| Privacy notice, English and Hindi, written to the DPDP Act's notice requirements | `site/privacy.html` | renders in both languages; placeholders marked `[TO FILL: …]` |
| Pre-launch check | `python tools/check_site.py` | fails on any missing link or file, an external script, a placeholder, or an unconnected form; `--draft` skips the last two |
| Hosted demo for a Hugging Face ZeroGPU Space | `space_omni/` (app, card, requirements), built by `python tools/build_space_omni.py` | run locally on CPU from the built folder: all 117 samples, 0 errors, scores as the local demo except Wikipedia lookup (2/10; its 1.3 GB index is not shipped) |

Not tested: the Space on real ZeroGPU hardware (it needs the Space created), and
the form against a real form service (it needs an account).

## To do before launch

Each needs a decision or an account that only the project's owners can make.

1. **Pick the form service and connect it.** Web3Forms (free, 250 submissions a
   month) or Tally (EU-hosted). Put its URL in `data-endpoint` on
   `<form id="enquiry-form">` in `site/index.html`, its public key in
   `data-access-key` if it uses one (Web3Forms does), and a fallback address in
   `data-email`.
2. **Fill the privacy notice.** Every `[TO FILL: …]` in `site/privacy.html`
   (English and Hindi): the organisation's legal name, the named contact person
   and email, the form service and the country it stores data in, the date.
   Have a lawyer read it once; the DPDP Rules' main duties apply from about
   May 2027, and the report lists what to confirm.
3. **Workshop prices** (optional). The page says "Fees: on request". The report
   found public AI workshops in India at about Rs 3,000-10,000 per seat.
4. **Domain and hosting.** Recommended: Cloudflare Pages for `site/` (free), a
   domain such as `anulm.org` from Cloudflare Registrar (about $8.50 the first
   year), GitHub Pages as a mirror.
5. **Create the Space.** `toonist` is a personal Hugging Face account, so it can
   host up to two ZeroGPU Spaces on the free plan. Build the folder, upload it,
   set the hardware to ZeroGPU in the Space's settings, then run
   `python experiments/demo_samples_check.py --url <the Space's URL>`. If ZeroGPU
   does not work for this app, the fallback is PRO ($9/month) plus CPU Upgrade
   ($0.03/hour, about $22/month), with no code change.
6. **Community.** Turn on GitHub Discussions for the repository (a setting;
   the page links to GitHub issues until then) and point the page's link at it.
7. **Run `python tools/check_site.py`** until it reports 0 problems, then deploy.

## After launch

From the report's plan: a GitHub release with a Zenodo DOI, a Hacker News "Show
HN" that links to the Space (not this page), an entry in AI4Bharat's IndicNLP
catalog, and one faculty partner for the arXiv endorsement and the academic
routes (AICTE ATAL faculty programmes for 2027-28, NPTEL Industry Associate,
Smart India Hackathon 2027).
