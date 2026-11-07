"""The intent label lists shipped in anulm/tasks/labels/ must equal what the data gives.

    python -m unittest tests.test_labels

A label list in the wrong order still classifies -- into the wrong names. The
Omni intent heads were trained against labels(task), so the packaged copy that
a fresh clone or a Space uses must match the data-derived one exactly.
"""
import json
import unittest

from anulm.tasks import decide


class PackagedLabels(unittest.TestCase):
    def test_lists_exist_and_are_unique(self):
        for task, n in (("banking77", 77), ("massive-hi", 60)):
            names = json.loads((decide.PACKAGED_LABELS / f"{task}.json").read_text(encoding="utf-8"))
            self.assertEqual(len(names), n, task)
            self.assertEqual(len(set(names)), n, task)

    def test_they_equal_the_data(self):
        for task, data in (("banking77", decide.DATA / "categories.json"),
                           ("massive-hi", decide.HERE / "data" / "decide_general" / "massive_hi-IN" / "train.jsonl")):
            if not data.exists():
                self.skipTest(f"{data} not on this machine")
            packaged = decide.labels(task)
            path = decide.PACKAGED_LABELS / f"{task}.json"
            saved = path.read_bytes()
            path.rename(path.with_suffix(".off"))
            try:
                from_data = decide.labels(task)
            finally:
                path.with_suffix(".off").rename(path)
            self.assertEqual(path.read_bytes(), saved)
            self.assertEqual(packaged, from_data, task)


if __name__ == "__main__":
    unittest.main()
