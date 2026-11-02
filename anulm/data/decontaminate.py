"""Exact Unicode-token overlap filtering against explicit local reference files.

No datasets are downloaded. This does not detect paraphrases or prove a corpus clean.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import uuid

from anulm.data.setup_multimodal import file_sha, save
from anulm.tokenization.bpe import DOC_SEP, iter_documents


def ngrams(text, n=13):
    if not isinstance(n, int) or n < 1:
        raise ValueError('n must be positive.')
    # Includes Unicode letters/marks and code punctuation; every window is scanned.
    tokens = re.findall(r'\w+|[^\w\s]', text.casefold(), flags=re.UNICODE)
    return {hashlib.sha256(json.dumps(tokens[i:i+n], ensure_ascii=False).encode()).digest()
            for i in range(len(tokens) - n + 1)}


def references(paths, fields, n):
    found, hashes = set(), {}
    for value in paths:
        path = Path(value).resolve()
        hashes[str(path)] = file_sha(path)
        if path.suffix.lower() == '.parquet':
            import pyarrow.parquet as pq
            rows = (row for batch in pq.ParquetFile(path).iter_batches(batch_size=256) for row in batch.to_pylist())
        else:
            rows = (json.loads(line) for line in path.read_text(encoding='utf-8').split('\n') if line.strip())
        for row in rows:
            for field in fields:
                text = row.get(field)
                if isinstance(text, list):
                    text = '\n'.join(map(str, text))
                if isinstance(text, str):
                    found.update(ngrams(text, n))
        if file_sha(path) != hashes[str(path)]:
            raise ValueError('Reference changed while reading.')
    if not found:
        raise ValueError('No usable reference n-grams. Filtering refused; check files, fields and n.')
    return found, hashes


def filter_corpus(corpus, reference_paths, output, fields, n=13):
    corpus, output = Path(corpus).resolve(), Path(output).resolve()
    if output == corpus or output in {Path(p).resolve() for p in reference_paths}:
        raise ValueError('Output must differ from all inputs.')
    if output.exists():
        raise FileExistsError('Choose a new output; existing corpora are not overwritten.')
    known, reference_hashes = references(reference_paths, fields, n)
    original = file_sha(corpus)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.' + uuid.uuid4().hex + '.tmp')
    kept = dropped = 0
    try:
        with temporary.open('w', encoding='utf-8', newline='\n') as stream:
            for doc in iter_documents(str(corpus)):
                if ngrams(doc, n) & known:
                    dropped += 1
                else:
                    stream.write(doc + DOC_SEP)
                    kept += 1
        if original != file_sha(corpus):
            raise ValueError('Corpus changed while filtering.')
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    report = {'algorithm': 'exact Unicode-token SHA256 windows', 'n': n, 'stride': 1,
        'corpus_sha256': original, 'reference_files': reference_hashes, 'fields': fields,
        'kept': kept, 'dropped': dropped, 'output_sha256': file_sha(output),
        'note': 'Exact overlap only; short matches and paraphrases are not covered. Retain input licenses.'}
    save(output.with_name(output.name + '.filter.json'), json.dumps(report, indent=2).encode())
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus', required=True)
    p.add_argument('--reference', nargs='+', required=True)
    p.add_argument('--fields', nargs='+', default=['prompt', 'text', 'code', 'canonical_solution', 'test', 'test_list'])
    p.add_argument('--out', required=True)
    p.add_argument('--n', type=int, default=13)
    a = p.parse_args()
    print(json.dumps(filter_corpus(a.corpus, a.reference, a.out, a.fields, a.n), indent=2))


if __name__ == '__main__':
    main()
