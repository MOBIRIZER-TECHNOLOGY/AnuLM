"""
AnuLM API demo: send every request in requests/, save each real response in
responses/, score the sample data in data/, and write TRANSCRIPT.md.

    python api.py --device cpu                       # terminal 1, from the repo root
    python examples/api_demo/run_demo.py             # terminal 2

    python examples/api_demo/run_demo.py --only decide      # requests whose name contains "decide"
    python examples/api_demo/run_demo.py --base-url http://10.0.0.5:8000 --key $ANULM_API_KEY
    python examples/api_demo/run_demo.py --only vision --image my_photo.jpg

Each response file holds the request as sent, the HTTP status, the time it took
and the body. Requests may point at local files with "@path" (relative to this
folder): an image_url becomes a base64 data URL, a multipart field is uploaded.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import time
from pathlib import Path

import requests

HERE = Path(__file__).parent
REQ, RESP, DATA = HERE / "requests", HERE / "responses", HERE / "data"


IMAGE: Path | None = None          # --image: use this picture instead of the one a request names


def data_url(path: Path) -> str:
    mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}[path.suffix.lower()]
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def resolve(obj):
    """Return a copy with "@path" image URLs replaced by data URLs."""
    if isinstance(obj, dict):
        out = {k: resolve(v) for k, v in obj.items()}
        if isinstance(out.get("url"), str) and out["url"].startswith("@"):
            out["url"] = data_url(IMAGE or HERE / out["url"][1:])
        return out
    if isinstance(obj, list):
        return [resolve(v) for v in obj]
    return obj


def shorten(obj, limit=120):
    """For the transcript: cut base64 blobs down to something readable."""
    if isinstance(obj, dict):
        return {k: shorten(v, limit) for k, v in obj.items()}
    if isinstance(obj, list):
        return [shorten(v, limit) for v in obj]
    if isinstance(obj, str) and obj.startswith("data:") and len(obj) > limit:
        return obj[:60] + f"...<{len(obj):,} chars>"
    return obj


def send(s: requests.Session, base: str, r: dict) -> dict:
    url = base + r["path"]
    t0 = time.time()
    if r["method"] == "GET":
        res = s.get(url, timeout=600)
    elif "multipart" in r:
        files, form = {}, {}
        for k, v in r["multipart"].items():
            if isinstance(v, str) and v.startswith("@"):
                p = HERE / v[1:]
                files[k] = (p.name, p.read_bytes())
            else:
                form[k] = v
        res = s.post(url, data=form, files=files, timeout=600)
    else:
        res = s.post(url, json=resolve(r["body"]), timeout=600, stream=bool(r["body"].get("stream")))
    out = {"status": res.status_code, "content_type": res.headers.get("content-type", "")}

    if r.get("body", {}).get("stream") and res.ok:
        events, text = [], ""
        for line in res.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                events.append("[DONE]")
                break
            ev = json.loads(payload)
            events.append(ev)
            for c in ev.get("choices", []):
                text += c.get("delta", {}).get("content") or ""
        out["body"] = {"assembled_text": text, "events": events}
    elif out["content_type"].startswith("application/json"):
        out["body"] = res.json()
    elif out["content_type"].startswith("text/"):
        out["body"] = res.text
    else:                                                    # audio
        name = r.get("save_as", "response.bin")
        (RESP / name).write_bytes(res.content)
        out["body"] = {"saved_to": f"responses/{name}", "bytes": len(res.content)}
    out["seconds"] = round(time.time() - t0, 3)
    return out


def score(s: requests.Session, base: str, model: str, csv_path: Path) -> dict:
    """Send every query in a labelled CSV to /v1/decide in one batch and score it."""
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    t0 = time.time()
    res = s.post(base + "/v1/decide", json={"model": model, "input": [r["query"] for r in rows],
                                             "threshold": 0.7, "top": 3}, timeout=600)
    res.raise_for_status()
    out = res.json()["results"]
    table = [{"query": r["query"], "expected": r["expected"], "choice": o["choice"],
              "confidence": o["confidence"], "action": o["action"], "correct": o["choice"] == r["expected"]}
             for r, o in zip(rows, out)]
    auto = [t for t in table if t["action"] == "auto-route"]
    return {"model": model, "data": f"data/{csv_path.name}", "n": len(table),
            "accuracy": round(sum(t["correct"] for t in table) / len(table), 3),
            "auto_routed": len(auto),
            "accuracy_when_auto_routed": round(sum(t["correct"] for t in auto) / len(auto), 3) if auto else None,
            "seconds": round(time.time() - t0, 3), "rows": table}


def curl(base: str, r: dict) -> str:
    url = base + r["path"]
    if r["method"] == "GET":
        return f"curl {url}"
    if "multipart" in r:
        parts = " ".join(f"-F {k}={v}" for k, v in r["multipart"].items())
        return f"curl {url} {parts}"
    body = json.dumps(shorten(r["body"]), ensure_ascii=False).replace("'", "'\\''")
    cmd = f"curl {url} -H \"Content-Type: application/json\" \\\n  -d '{body}'"
    if '"@' in body:                                         # run_demo.py inlines the file; curl cannot
        cmd = "# replace \"@data/...\" with a data URL: data:image/jpeg;base64,$(base64 -w0 data/astronaut.jpg)\n" + cmd
    return cmd


def transcript(base: str, runs: list, evals: list) -> str:
    L = ["# AnuLM API: real requests and responses", "",
         f"Recorded {time.strftime('%Y-%m-%d %H:%M')} against `{base}` by `run_demo.py`.",
         "Every block below is what the server actually returned; ids, timings and sampled",
         "text change from run to run. Long base64 strings are shortened.", ""]
    for name, r, out in runs:
        body = out["body"]
        if name == "01_list_models":                         # 10 long descriptions: keep the gist
            body = {"object": "list", "data": [{"id": m["id"], "from_scratch": m.get("from_scratch"),
                                                "endpoints": m.get("endpoints")} for m in body.get("data", [])]}
        elif name == "04_chat_stream" and isinstance(body, dict) and "events" in body:
            ev = body["events"]
            body = {"assembled_text": body["assembled_text"],
                    "events": ev[:2] + [f"... {len(ev) - 4} more chunks ..."] + ev[-2:] if len(ev) > 6 else ev}
        L += [f"## {name[:2]}. {r['title']}", "", f"`{r['method']} {r['path']}` → **{out['status']}** "
              f"in {out['seconds']} s", "", "```bash", curl(base, r), "```", ""]
        if "body" in r:
            L += ["Request:", "", "```json", json.dumps(shorten(r["body"]), ensure_ascii=False, indent=2), "```", ""]
        L += ["Response:", "", "```json", json.dumps(shorten(body), ensure_ascii=False, indent=2), "```", ""]
    if evals:
        L += ["## Sample data scored through /v1/decide", ""]
        for e in evals:
            auto = (f"{e['auto_routed']}/{e['n']} auto-routed at {e['accuracy_when_auto_routed']:.0%}"
                    if e["auto_routed"] else "none auto-routed")
            L += [f"### `{e['model']}` on `{e['data']}`: {e['accuracy']:.0%} of {e['n']} ({auto})", "",
                  "| query | expected | choice | confidence | action |", "|---|---|---|---|---|"]
            for t in e["rows"]:
                mark = "" if t["correct"] else " ✗"
                L.append(f"| {t['query']} | {t['expected']} | {t['choice']}{mark} | {t['confidence']:.2f} "
                         f"| {t['action']} |")
            L.append("")
    return "\n".join(L)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--key", default=os.environ.get("ANULM_API_KEY"))
    p.add_argument("--only", default="", help="run only requests whose file name contains this")
    p.add_argument("--no-eval", action="store_true", help="skip scoring the CSVs")
    p.add_argument("--image", type=Path, help="a .jpg or .png to send to the vision requests")
    a = p.parse_args()
    global IMAGE
    IMAGE = a.image
    base = a.base_url.rstrip("/")
    RESP.mkdir(exist_ok=True)

    s = requests.Session()
    if a.key:
        s.headers["Authorization"] = f"Bearer {a.key}"
    try:
        s.get(base + "/health", timeout=5).raise_for_status()
    except requests.RequestException as e:
        raise SystemExit(f"No AnuLM API at {base} ({e}). Start it with: python api.py --device cpu")

    runs = []
    for f in sorted(REQ.glob("*.json")):
        if a.only not in f.stem:
            continue
        r = json.loads(f.read_text(encoding="utf-8"))
        try:
            out = send(s, base, r)
        except Exception as e:                               # keep going; record what broke
            out = {"status": None, "seconds": None, "body": {"client_error": repr(e)}}
        (RESP / f.name).write_text(json.dumps({"request": shorten(r), **out}, ensure_ascii=False, indent=2)
                                   + "\n", encoding="utf-8")
        runs.append((f.stem, r, out))
        print(f"{f.stem:28s} {out['status']}  {out['seconds']} s")

    evals = []
    if not a.no_eval and not a.only:
        for model, name in (("anulm-decide-banking", "banking_queries.csv"),
                            ("anulm-decide-hindi", "hindi_queries.csv")):
            try:
                e = score(s, base, model, DATA / name)
            except Exception as ex:
                print(f"eval {model}: failed ({ex})")
                continue
            evals.append(e)
            (RESP / f"eval_{model}.json").write_text(json.dumps(e, ensure_ascii=False, indent=2) + "\n",
                                                     encoding="utf-8")
            print(f"eval {model:24s} {e['accuracy']:.0%} of {e['n']}, {e['auto_routed']} auto-routed")
        answers = []
        for q in (DATA / "questions.txt").read_text(encoding="utf-8").split("\n"):
            if q.strip():
                res = s.post(base + "/v1/answer", json={"question": q.strip()}, timeout=600)
                answers.append(res.json())
        (RESP / "eval_questions.json").write_text(json.dumps(answers, ensure_ascii=False, indent=2) + "\n",
                                                  encoding="utf-8")
        print(f"answered {len(answers)} questions -> responses/eval_questions.json")

    if not a.only:
        (HERE / "TRANSCRIPT.md").write_text(transcript(base, runs, evals), encoding="utf-8")
        print("wrote", HERE / "TRANSCRIPT.md")


if __name__ == "__main__":
    main()
