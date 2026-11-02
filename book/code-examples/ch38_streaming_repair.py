"""Book extension: byte-stream repair reference, not the historical API.

    python book/code-examples/ch38_streaming_repair.py

No model, checkpoint or third-party package is needed. Integrating a different
tokenizer requires passing its actual token bytes, not re-encoding incomplete
strings returned by a lossy decoder.
"""
from __future__ import annotations

import codecs


class TextStream:
    """Release complete UTF-8 text while holding possible stop-string prefixes."""

    def __init__(self, stops=()):
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
        self.stops = tuple(s for s in stops if s)
        self.pending = ""
        self.text = ""
        self.stopped = False
        self.finished = False
        self.dropped_incomplete_bytes = 0

    def _release(self, final=False):
        matches = [self.pending.find(s) for s in self.stops if s in self.pending]
        if matches:
            count = min(matches)
            self.stopped = True
        else:
            hold = 0 if final else max((len(s) - 1 for s in self.stops), default=0)
            count = max(0, len(self.pending) - hold)
        delta = self.pending[:count]
        self.pending = "" if self.stopped else self.pending[count:]
        self.text += delta
        return delta

    def feed(self, data: bytes):
        if self.finished:
            raise ValueError("Cannot feed a finished stream")
        if self.stopped:
            return ""
        self.pending += self.decoder.decode(data, final=False)
        return self._release()

    def finish(self):
        if self.finished:
            return ""
        self.finished = True
        # An incomplete trailing code point is deliberately omitted, not turned
        # into U+FFFD. A genuine, fully encoded U+FFFD remains ordinary text.
        self.dropped_incomplete_bytes = len(self.decoder.getstate()[0])
        return "" if self.stopped else self._release(final=True)


def scripted_reply(tokens, stops=(), max_tokens=1000, eos=257):
    stream = TextStream(stops)
    emitted = []

    def emit(delta):
        if delta:
            emitted.append(delta)

    for token in tokens[:max_tokens]:
        if token == eos:
            emit(stream.finish())  # Both update returned text AND send the final delta.
            break
        emit(stream.feed(bytes([token])))
        if stream.stopped:
            break
    emit(stream.finish())  # Idempotent; handles a token limit or a stopped stream.
    assert "".join(emitted) == stream.text
    return stream.text, "".join(emitted), stream.dropped_incomplete_bytes


def main():
    cases = [
        ("EOS with held suffix", list(b"Hello") + [257], ["###"], 1000, "Hello"),
        ("EOS with unequal stop lengths", list(b"Hello") + [257], ["\n\n", "User:"], 1000, "Hello"),
        ("stop across tokens", list(b"Hi.\n\nNext"), ["\n\n"], 1000, "Hi."),
        ("complete Hindi", list("नमस्ते".encode()), [], 1000, "नमस्ते"),
        ("limit inside UTF-8", list("Hi न".encode()), [], 5, "Hi "),
        ("genuine replacement character", list("Hi �".encode()) + [257], [], 1000, "Hi �"),
        ("short reply with long stop", list(b"H") + [257], ["User:"], 1000, "H"),
    ]
    for name, tokens, stops, limit, expected in cases:
        returned, sent, dropped = scripted_reply(tokens, stops, limit)
        assert returned == sent == expected, (name, returned, sent, expected)
        print(f"PASS: {name}; returned and streamed text agree")
        if name == "limit inside UTF-8":
            assert dropped == 2
    print("7 checks passed; this is a byte-mode repair reference, not a patched live server.")


if __name__ == "__main__":
    main()
