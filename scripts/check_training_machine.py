"""Check the destination machine without downloading data or starting training."""
import json
from pathlib import Path
import shutil
import sys

import torch
import torchvision
import timm
import faster_whisper
import pyarrow
import soundfile
import tokenizers
from playwright.sync_api import sync_playwright

root = Path(__file__).resolve().parents[1]
assert shutil.which('git'), 'Install Git and reopen your terminal so git is on PATH.'
assert sys.version_info[:2] == (3, 14), 'This setup recipe targets Python 3.14.'
assert torch.cuda.is_available(), 'CUDA unavailable. Check the NVIDIA driver and CUDA PyTorch installation.'
assert torch.__version__.split('+')[0] == '2.14.0', 'Unexpected torch version.'
assert torchvision.__version__.split('+')[0] == '0.29.0', 'Unexpected torchvision version.'
free = shutil.disk_usage(root).free / 1024**3
assert free >= 150, f'Only {free:.1f} GiB free. Allow at least 150 GiB for this full-data run.'
x = torch.ones(16, device='cuda')
assert (x @ x).item() == 16
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.set_content('<p>Rendering check</p>')
    assert page.locator('p').inner_text() == 'Rendering check'
    browser.close()
print(json.dumps({'status': 'passed', 'python': sys.version, 'torch': torch.__version__,
                  'gpu': torch.cuda.get_device_name(0),
                  'gpu_bytes': torch.cuda.get_device_properties(0).total_memory,
                  'free_disk_gib': round(free, 1)}, indent=2))
print('Machine setup check only; real multimodal preflight runs before training.')
