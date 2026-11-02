"""Clearly marked procedural Indian-context price cards and arithmetic QA.

These are controlled synthetic tasks, not real photos or representative Indian
conversations. Ground truth is computed directly; validation/test groups are
assigned before rendering or translation.
"""
from __future__ import annotations

import hashlib
import base64
from contextlib import contextmanager
import json
from pathlib import Path

from anulm.data.india_corpus import OUT, download
from anulm.data.setup_multimodal import jsonl, save

FONT_REVISION = '2bc5d431b584d89e646214bf6cfa75494e76ceea'
FONT_BASE = f'https://raw.githubusercontent.com/google/fonts/{FONT_REVISION}/ofl/notosansdevanagari/'
ITEMS = [('rice', 'चावल'), ('milk', 'दूध'), ('tea', 'चाय'), ('sugar', 'चीनी'),
         ('salt', 'नमक'), ('flour', 'आटा'), ('soap', 'साबुन'), ('oil', 'तेल')]


@contextmanager
def renderer(font):
    """Chromium shapes Devanagari correctly even without Pillow/libraqm."""
    from playwright.sync_api import sync_playwright
    candidates = [Path(r'C:\Program Files\Google\Chrome\Application\chrome.exe'),
                  Path(r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe')]
    executable = next((str(path) for path in candidates if path.exists()), None)
    with sync_playwright() as engine:
        browser = engine.chromium.launch(executable_path=executable, headless=True)
        try:
            page = browser.new_page(viewport={'width': 224, 'height': 224}, device_scale_factor=1)
            encoded = base64.b64encode(font.read_bytes()).decode('ascii')
            page.set_content('''<style>@font-face{font-family:Card;src:url(data:font/ttf;base64,'''
                + encoded + ''')}*{box-sizing:border-box}body{margin:0;background:#f8f4e6;color:#0f1923;
                font:29px Card}main{position:absolute;inset:8px;border:3px solid #283c46;padding:20px 7px}
                #price{margin-top:36px}</style><main><div id="item"></div><div id="price"></div></main>''')
            page.evaluate('document.fonts.ready')
            def render(label, price, path):
                page.evaluate('(v)=>{document.querySelector("#item").textContent=v[0];document.querySelector("#price").textContent="INR "+v[1]}', [label, price])
                page.screenshot(path=str(path), animations='disabled')
            yield render
        finally:
            browser.close()


def partition(key):
    bucket = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 100
    return 'validation' if bucket < 3 else 'test' if bucket < 6 else 'train'


def generate(out=OUT, count=6000):
    font = out / 'fonts/NotoSansDevanagari.ttf'
    download(FONT_BASE + 'NotoSansDevanagari%5Bwdth,wght%5D.ttf', font)
    download(FONT_BASE + 'OFL.txt', out / 'licenses/NotoSansDevanagari-OFL.txt')
    captions = {split: [] for split in ['train', 'validation', 'test']}
    questions = {split: [] for split in captions}
    text = {split: [] for split in captions}
    seen_questions = set()
    media = out / 'media/india-synthetic'
    media.mkdir(parents=True, exist_ok=True)
    with renderer(font) as render:
        for index in range(count):
            group_index = index // 2
            item_index = group_index % len(ITEMS)
            english, hindi = ITEMS[item_index]
            price = group_index // len(ITEMS) + 1
            split = partition(f'price:{item_index}:{price}')
            language = 'hi' if index % 2 else 'en'
            label = hindi if language == 'hi' else english
            path = media / f'price-{index:05d}.png'
            render(label, price, path)
            caption = (f'{hindi} की कीमत {price} रुपये है।' if language == 'hi'
                       else f'A price card for {english} showing INR {price}.')
            common = {'image': str(path.resolve()), 'source': 'project-procedural-price-cards',
                      'synthetic': True, 'group': f'price:{item_index}:{price}',
                      'font_source': FONT_BASE, 'font_license': 'OFL-1.1', 'lang': language}
            captions[split].append({**common, 'text': caption})
            question = 'कीमत कितने रुपये है?' if language == 'hi' else 'What is the price in rupees?'
            questions[split].append({**common, 'question': question, 'text': str(price), 'answers': [str(price)]})
            quantity = index % 5 + 1
            question = (f'एक वस्तु की कीमत {price} रुपये है। {quantity} वस्तुओं की कुल कीमत क्या है?'
                        if language == 'hi' else f'An item costs INR {price}. What do {quantity} items cost?')
            math_group = f'arithmetic:{price}:{quantity}'
            math_split = partition(math_group)
            if question not in seen_questions:
                seen_questions.add(question)
                text[math_split].append({'question': question, 'answer': str(price * quantity), 'lang': language,
                                        'source': 'project-procedural-integer-arithmetic', 'synthetic': True,
                                        'group': math_group})
    counts = {}
    for name, parts in [('india_caption', captions), ('india_vqa', questions), ('india_qa', text)]:
        for split, records in parts.items():
            jsonl(out / f'{name}_{split}.jsonl', records)
            counts[f'{name}_{split}'] = len(records)
    save(out / 'SYNTHETIC.json', json.dumps({'counts': counts,
        'scope': 'Controlled Hindi/English price-card OCR, captions, arithmetic; not real-world representativeness.',
        'font_revision': FONT_REVISION}, indent=2).encode())
    return counts


if __name__ == '__main__':
    print(generate())
