"""Drive the Gradio demo in headless Chrome and photograph it.

    python app.py --port 7860 &
    python tools_screenshot.py docs/media/demo-coder.png

Why this exists rather than "take a screenshot": the image in
`docs/ANNOUNCEMENT.md` shows a real model answering a real prompt, and it has
to stay true when the page changes. This drives the page the way a visitor
does -- picks a checkpoint, presses Load, types, presses Generate, waits for
output -- and only then captures it. If the interface breaks, this fails
loudly instead of producing a pretty picture of a broken page.

It is also the closest thing here to a visual test. The functional tests in
`test_model.py` drive the app's Python; this drives its DOM, which is where
"the examples never changed when you switched model" was eventually found.

No new dependency: Chrome speaks the DevTools Protocol over a websocket, and
`websockets` comes with gradio. Chrome itself is found in the usual places or
named with --chrome.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

CHROMES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]


def find_chrome(explicit: str | None) -> str:
    for c in ([explicit] if explicit else []) + CHROMES:
        if c and Path(c).exists():
            return c
    raise SystemExit("no chrome found; pass --chrome /path/to/chrome")


async def shoot(args) -> int:
    proc = subprocess.Popen(
        [find_chrome(args.chrome), "--headless=new",
         f"--remote-debugging-port={args.debug_port}",
         f"--window-size={args.width},1600", "--hide-scrollbars", "--disable-gpu",
         "--no-first-run", "--no-default-browser-check", "--incognito", args.url],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        ws_url = None
        for _ in range(40):
            try:
                tabs = json.load(urllib.request.urlopen(
                    f"http://127.0.0.1:{args.debug_port}/json"))
                pages = [t for t in tabs if t["type"] == "page"]
                if pages:
                    ws_url = pages[0]["webSocketDebuggerUrl"]
                    break
            except Exception:
                pass
            time.sleep(0.5)
        if not ws_url:
            print("chrome's debugger never answered"); return 1

        import websockets
        async with websockets.connect(ws_url, max_size=40 * 1024 * 1024) as ws:
            seq = 0

            async def cmd(method, **params):
                nonlocal seq
                seq += 1
                await ws.send(json.dumps({"id": seq, "method": method, "params": params}))
                while True:
                    msg = json.loads(await ws.recv())
                    if msg.get("id") == seq:
                        return msg.get("result", {})

            async def js(expr, wait=0.0):
                r = await cmd("Runtime.evaluate", expression=expr,
                              awaitPromise=True, returnByValue=True)
                if wait:
                    await asyncio.sleep(wait)
                return r.get("result", {}).get("value")

            async def click(label):
                return await js(f"""
                    (() => {{
                      const b = [...document.querySelectorAll('button')]
                          .find(x => x.textContent.trim() === {label!r});
                      if (!b) return 'missing';
                      b.click(); return 'clicked';
                    }})()""", wait=1)

            await cmd("Page.enable")
            await cmd("Runtime.enable")
            await asyncio.sleep(args.boot)
            assert await js("document.title") == "AnuLM", "this is not the AnuLM page"

            assert await click("Load") == "clicked", "no Load button"
            for _ in range(args.timeout):
                if await js("document.body.innerText.includes('Loaded —')"):
                    break
                await asyncio.sleep(1)
            else:
                print("the model never finished loading"); return 1

            typed = await js(f"""
                (() => {{
                  const t = document.querySelector('textarea');
                  if (!t) return 'missing';
                  const set = Object.getOwnPropertyDescriptor(
                      window.HTMLTextAreaElement.prototype, 'value').set;
                  set.call(t, {args.prompt!r});
                  t.dispatchEvent(new Event('input', {{bubbles: true}}));
                  return 'typed';
                }})()""", wait=1)
            assert typed == "typed", "no prompt box"
            assert await click("Generate") == "clicked", "no Generate button"

            for _ in range(args.timeout):
                out = await js("""
                    (() => { const ts = [...document.querySelectorAll('textarea')];
                             return ts.length > 1 ? ts[ts.length - 1].value : ''; })()""")
                if out and out.strip():
                    print("generated:", out[:60].replace("\n", " / "))
                    break
                await asyncio.sleep(1)
            else:
                print("nothing was generated"); return 1

            height = await js("Math.max(document.body.scrollHeight, 900)")
            await cmd("Emulation.setDeviceMetricsOverride", width=args.width,
                      height=int(min(height, 2600)), deviceScaleFactor=args.scale,
                      mobile=False)
            await asyncio.sleep(1)
            shot = await cmd("Page.captureScreenshot", format="png",
                             captureBeyondViewport=True)
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_bytes(base64.b64decode(shot["data"]))
            print(f"saved {args.out} ({os.path.getsize(args.out) / 1024:.0f} kB)")
            return 0
    finally:
        proc.terminate()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", nargs="?", default="docs/media/demo-coder.png")
    p.add_argument("--url", default="http://127.0.0.1:7860/")
    p.add_argument("--prompt",
                   default="Write a Python function to check whether a number is prime.")
    p.add_argument("--chrome", help="path to chrome, if it is somewhere unusual")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--scale", type=float, default=2, help="device pixel ratio")
    p.add_argument("--boot", type=float, default=6, help="seconds for gradio's front end")
    p.add_argument("--timeout", type=int, default=90, help="seconds to wait for the model")
    p.add_argument("--debug-port", type=int, default=9222)
    return asyncio.run(shoot(p.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
