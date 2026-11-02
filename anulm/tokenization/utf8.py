"""Decode complete UTF-8 characters, retaining a valid literal U+FFFD."""
import codecs


def complete_utf8(data):
    return codecs.getincrementaldecoder('utf-8')(errors='replace').decode(data, final=False)
