"""Regression tests for the Wikipedia lookup: rag.py's title rerank, the
reader's retry and citation (omni.Reader), and `rag build-dense`.

No network, no GPU, no weights: the encoder and the model are stand-ins.

    python -m unittest tests.test_quality_retrieval
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from anulm.inference import rag


class TitleMatchTests(unittest.TestCase):
    def test_the_article_the_question_names(self):
        self.assertEqual(rag.title_match("Hamlet", "Who wrote Hamlet?"), 1)
        self.assertEqual(rag.title_match("Taj Mahal", "Who built the Taj Mahal?"), 1)
        self.assertEqual(rag.title_match("ताजमहल", "ताजमहल किसने बनवाया था?"), 1)

    def test_side_pages_go_down(self):
        # Bug: "Who wrote Hamlet?" was answered from "Hamlet (1964 movie)".
        self.assertEqual(rag.title_match("Hamlet (1964 movie)", "Who wrote Hamlet?"), -1)
        self.assertEqual(rag.title_match("List of planets", "What is the largest planet?"), -1)
        self.assertEqual(rag.title_match("रामायण (बहुविकल्पी)", "रामायण किसने लिखी?"), -1)
        self.assertEqual(rag.title_match("भारतीय राज्य पशुओं की सूची", "भारत का राष्ट्रीय पशु कौन सा है?"), -1)

    def test_a_qualifier_the_question_names_is_kept(self):
        self.assertEqual(rag.title_match("Mercury (planet)", "How hot is the planet Mercury?"), 1)

    def test_only_names_count_in_latin_script(self):
        # "planet" is a common noun here; "Solar System" is capitalised, a name.
        q = "What is the largest planet in the Solar System?"
        self.assertEqual(rag.title_match("Planet", q), 0)
        self.assertEqual(rag.title_match("Solar System", q), 1)

    def test_whole_words_only(self):
        self.assertEqual(rag.title_match("Ham", "Who wrote Hamlet?"), 0)
        self.assertEqual(rag.title_match("India", "Who was the first Prime Minister of Indiana?"), 0)


class RerankTests(unittest.TestCase):
    def hit(self, title, score):
        return {"title": title, "text": "", "score": score}

    def test_near_tie_goes_to_the_named_article(self):
        hits = [self.hit("Hamlet (1964 movie)", 0.870), self.hit("Hamlet", 0.866), self.hit("Hamnet", 0.847)]
        self.assertEqual([h["title"] for h in rag.rerank("Who wrote Hamlet?", hits, 0.01)],
                         ["Hamlet", "Hamlet (1964 movie)", "Hamnet"])

    def test_a_clear_lead_is_not_overturned(self):
        hits = [self.hit("Penicillin", 0.895), self.hit("Fleming", 0.860)]
        out = rag.rerank("Who discovered penicillin? Ask Fleming", hits, 0.01)
        self.assertEqual(out[0]["title"], "Penicillin")

    def test_off_by_default_keeps_dense_order(self):
        # Measured and left off (rag.TITLE_BONUS): the default must not reorder.
        hits = [self.hit("Hamlet (1964 movie)", 0.870), self.hit("Hamlet", 0.866)]
        self.assertEqual(rag.rerank("Who wrote Hamlet?", hits), hits)

    def test_ties_keep_their_order(self):
        hits = [self.hit("A", 0.5), self.hit("B", 0.5), self.hit("C", 0.5)]
        self.assertEqual([h["title"] for h in rag.rerank("x?", hits, 0.01)], ["A", "B", "C"])


class AnswerCheckTests(unittest.TestCase):
    def test_echoes(self):
        self.assertTrue(rag.echoes("Hamlet", "Who wrote Hamlet?"))
        self.assertTrue(rag.echoes("The Taj Mahal", "Who built the Taj Mahal?"))
        self.assertTrue(rag.echoes("भारत का राष्ट्रीय पशु", "भारत का राष्ट्रीय पशु कौन सा है?"))
        self.assertTrue(rag.echoes("", "Who?"))
        self.assertFalse(rag.echoes("William Shakespeare", "Who wrote Hamlet?"))

    def test_cite_names_the_passage_with_the_answer(self):
        hits = [{"title": "Hamlet (1964 movie)", "text": "A Soviet drama movie."},
                {"title": "Hamlet", "text": "A tragedy by William Shakespeare."}]
        self.assertEqual(rag.cite("William Shakespeare", hits), "Hamlet")
        self.assertEqual(rag.cite("Shakespear", hits), "Hamlet (1964 movie)")   # whole words; else the top hit


class ReaderRetryTests(unittest.TestCase):
    def reader(self, replies, hits):
        from anulm.multimodal.omni import Reader
        prompts = []

        def generate(prompt, *a, **k):
            prompts.append(prompt)
            return {"completion": " " + replies[len(prompts) - 1] + " "}
        om = SimpleNamespace(device="cpu", engine=SimpleNamespace(generate=generate))
        r = SimpleNamespace(search=lambda q, k: hits)
        return Reader(om, r), prompts

    HITS = [{"title": "Hamlet (1964 movie)", "text": "A Soviet drama movie."},
            {"title": "Hamlet", "text": "A tragedy by William Shakespeare."},
            {"title": "Hamnet", "text": "Shakespeare's son."}]

    def test_an_echo_is_read_again_rotated(self):
        rd, prompts = self.reader(["Hamlet", "William Shakespeare"], self.HITS)
        out = rd.ask("Who wrote Hamlet?")
        self.assertEqual(out["answer"], "William Shakespeare")
        self.assertEqual(len(prompts), 2)
        self.assertIn("[1] A tragedy by William Shakespeare.", prompts[1])     # passage 2 now first
        self.assertEqual(out["sources"], ["Hamlet", "Hamlet (1964 movie)", "Hamnet"])

    def test_nothing_found_is_empty_so_the_caller_falls_back(self):
        rd, prompts = self.reader(["not found in the passages"] * 3, self.HITS)
        self.assertEqual(rd.ask("Who wrote Hamlet?")["answer"], "")
        self.assertEqual(len(prompts), 3)

    def test_a_good_first_answer_reads_once(self):
        rd, prompts = self.reader(["William Shakespeare"], self.HITS)
        self.assertEqual(rd.ask("Who wrote Hamlet?")["answer"], "William Shakespeare")
        self.assertEqual(len(prompts), 1)


def tiny_index(root: Path, n_chunks: int, dense_rows: int | None) -> None:
    """The files Retriever opens, for `n_chunks` chunks (BM25 arrays empty)."""
    root.mkdir(parents=True, exist_ok=True)
    with open(root / "chunks.jsonl", "w", encoding="utf-8") as f:
        for i in range(n_chunks):
            f.write(json.dumps({"lang": "en", "title": f"T{i}", "text": f"text {i}"}) + "\n")
    for name in ("indptr", "t_indptr"):
        np.save(root / f"{name}.npy", np.zeros(2, np.int64))
    for name in ("docs", "t_docs", "chunk_art", "chunk_pos"):
        np.save(root / f"{name}.npy", np.zeros(n_chunks if name.startswith("chunk") else 0, np.int32))
    np.save(root / "weights.npy", np.zeros(0, np.float16))
    np.save(root / "idf.npy", np.zeros(1, np.float32))
    np.save(root / "t_len.npy", np.ones(n_chunks, np.float32))
    if dense_rows is not None:
        np.save(root / "dense_e5s.npy", np.zeros((dense_rows, 4), np.float16))


class DenseIndexTests(unittest.TestCase):
    def test_retriever_refuses_a_dense_index_of_the_wrong_size(self):
        # A --limit trial file in data/rag would silently map rows to the wrong passages.
        with tempfile.TemporaryDirectory() as d:
            tiny_index(Path(d), 5, 3)
            with self.assertRaisesRegex(ValueError, "build-dense"):
                rag.Retriever(Path(d), "cpu")
            tiny_index(Path(d), 5, 5)
            self.assertEqual(rag.Retriever(Path(d), "cpu").mode, "dense")

    def test_build_dense_embeds_chunks_in_order_to_out(self):
        built = {}

        class FakeDense:
            def __init__(self, device, index):
                built["device"], built["index"] = device, index

            def build(self, chunks, batch):
                built["titles"], built["batch"] = [c["title"] for c in chunks], batch
        with tempfile.TemporaryDirectory() as d:
            root, e5 = Path(d) / "rag", Path(d) / "e5"
            tiny_index(root, 5, None)
            e5.mkdir()
            (e5 / "model.safetensors").write_bytes(b"")       # present: nothing is downloaded
            with patch.object(rag, "Dense", FakeDense), patch.object(rag, "E5", e5):
                out = rag.build_dense(root, Path(d) / "x.npy", limit=3, device="cpu")
        self.assertEqual(out, Path(d) / "x.npy")
        self.assertEqual(built, {"device": "cpu", "index": Path(d) / "x.npy",
                                 "titles": ["T0", "T1", "T2"], "batch": 64})


if __name__ == "__main__":
    unittest.main()
