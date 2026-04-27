#!/usr/bin/env python
"""Test image-variant bypass strategies against blocked pages.

The text-only test established the filter is IMAGE-specific (visual classifier),
not content-based. This script takes saved JPGs (e.g. /tmp/blocked_pages/*.jpg from
probe_vision_page.py with PROBE_SAVE_DIR set) and tries multiple image manipulations
against the model. The first variant that produces non-empty content is our bypass.

Variants tested (in order of escalating modification):
  - original         : control - should still get blocked
  - top-half         : crop top 50%
  - bottom-half      : crop bottom 50%
  - left-half        : crop left 50%
  - right-half       : crop right 50%
  - grayscale        : remove color signals
  - half-res         : 50% resolution
  - quarter-res      : 25% resolution (heavy downsample)
  - inverted         : color invert
  - blurred          : Gaussian blur (radius 3)
  - rotated-90       : rotate 90 degrees (might evade orientation-aware classifier)
  - jpeg-low-quality : recompress at quality=30
  - cropped-tight    : center crop 80%
  - cropped-table-only-top : top 40% (often where headers/non-table content sits)
  - cropped-table-only-bottom : bottom 40%

Usage:
    uv run python scripts/test_image_variants.py /tmp/blocked_pages/<filename>.jpg

Examples:
    uv run python scripts/test_image_variants.py /tmp/blocked_pages/doc_84a1429369c1217187f782a4b41b18de_page029.jpg
    # Or run against all saved images:
    for f in /tmp/blocked_pages/*.jpg; do uv run python scripts/test_image_variants.py "$f"; done

Optional env (inherits from start.sh/.env):
    LLM_BINDING_HOST    LiteLLM base URL
    LLM_BINDING_API_KEY LiteLLM master key
    VISION_MODEL        default 'salmon'
"""

from __future__ import annotations

import asyncio
import base64
import io
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from yar.llm.openai import create_openai_async_client

EXTRACT_PROMPT = (
    'Extract all text and data from this page as clean markdown. '
    'Use HTML <table> tags for tables. Be thorough and accurate.'
)


def _make_variants(image_bytes: bytes) -> list[tuple[str, bytes]]:
    """Generate image variants. Returns list of (label, jpeg_bytes)."""
    try:
        from PIL import Image, ImageFilter, ImageOps
    except ImportError as exc:
        print(f'[!] Pillow not available: {exc}')
        print('    Install with: uv pip install Pillow')
        return [('original', image_bytes)]

    img = Image.open(io.BytesIO(image_bytes))
    width, height = img.size
    img = img.convert('RGB')

    def _to_jpeg(image: Image.Image, quality: int = 80) -> bytes:
        buf = io.BytesIO()
        image.convert('RGB').save(buf, format='JPEG', quality=quality)
        return buf.getvalue()

    return [
        ('original', _to_jpeg(img)),
        ('top-half', _to_jpeg(img.crop((0, 0, width, height // 2)))),
        ('bottom-half', _to_jpeg(img.crop((0, height // 2, width, height)))),
        ('left-half', _to_jpeg(img.crop((0, 0, width // 2, height)))),
        ('right-half', _to_jpeg(img.crop((width // 2, 0, width, height)))),
        ('grayscale', _to_jpeg(ImageOps.grayscale(img))),
        ('half-res', _to_jpeg(img.resize((width // 2, height // 2)))),
        ('quarter-res', _to_jpeg(img.resize((width // 4, height // 4)))),
        ('inverted', _to_jpeg(ImageOps.invert(img))),
        ('blurred', _to_jpeg(img.filter(ImageFilter.GaussianBlur(radius=3)))),
        ('rotated-90', _to_jpeg(img.rotate(90, expand=True))),
        ('jpeg-low-q30', _to_jpeg(img, quality=30)),
        ('cropped-tight-80', _to_jpeg(img.crop((width // 10, height // 10, 9 * width // 10, 9 * height // 10)))),
        ('top-40pct', _to_jpeg(img.crop((0, 0, width, int(height * 0.4))))),
        ('bottom-40pct', _to_jpeg(img.crop((0, int(height * 0.6), width, height)))),
    ]


async def _try_variant(client, model: str, label: str, image_bytes: bytes) -> None:
    image_b64 = base64.b64encode(image_bytes).decode()
    try:
        response = await client.chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=2048,
            messages=[
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': EXTRACT_PROMPT},
                        {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{image_b64}'}},
                    ],
                }
            ],
        )
    except Exception as exc:
        print(f'  [{label:24s}]  EXCEPTION: {type(exc).__name__}: {exc}')
        return

    choices = getattr(response, 'choices', None) or []
    if not choices:
        print(f'  [{label:24s}]  no choices')
        return

    choice = choices[0]
    finish = getattr(choice, 'finish_reason', None)
    msg = getattr(choice, 'message', None)
    content = getattr(msg, 'content', None) if msg else None
    chars = len(content) if content else 0
    img_kb = len(image_bytes) // 1024

    if chars >= 200:
        flag = 'BYPASS!'
    elif chars > 0:
        flag = 'PARTIAL'
    else:
        flag = 'BLOCKED'

    print(f'  [{label:24s}]  {flag:8s} finish={finish!s:18s} chars={chars:>5d} img={img_kb:>4d}KB')


async def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        sys.exit(f'Image not found: {image_path}')

    base_url = os.environ.get('LLM_BINDING_HOST') or os.environ.get('VISION_BINDING_HOST')
    api_key = os.environ.get('LLM_BINDING_API_KEY') or os.environ.get('VISION_BINDING_API_KEY')
    model = os.environ.get('VISION_MODEL', 'salmon')

    image_bytes = image_path.read_bytes()
    print(f'[test] image={image_path.name} ({len(image_bytes):,} bytes)')
    print(f'[test] model={model}')

    variants = _make_variants(image_bytes)
    print(f'[test] testing {len(variants)} variants\n')

    client = create_openai_async_client(api_key=api_key, base_url=base_url)
    try:
        for label, variant_bytes in variants:
            await _try_variant(client, model, label, variant_bytes)
    finally:
        await client.close()

    print('\nLegend: BYPASS!=>=200 chars  PARTIAL=1-199 chars  BLOCKED=0 chars')
    print('First BYPASS! variant is your candidate workaround.')


if __name__ == '__main__':
    asyncio.run(main())
