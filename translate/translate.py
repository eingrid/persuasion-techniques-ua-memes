import asyncio
import json
import os
import textwrap
from pathlib import Path

import cv2
import easyocr
import numpy as np
from openai import AsyncOpenAI
from PIL import Image, ImageDraw, ImageFont
from simple_lama_inpainting import SimpleLama
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "datasets" / "propaganda_950"
ANNOTATIONS_DIR = DATA_ROOT / "annotations"
IMAGE_ROOT = DATA_ROOT / "images"

OUT_ROOT = REPO_ROOT / "datasets" / "translated"
OUT_IMAGES_NO_TEXT = OUT_ROOT / "images_no_text"
OUT_IMAGES_TEXT = OUT_ROOT / "images"
OUT_ANNOTATIONS = OUT_ROOT / "annotations"

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

MODEL = "gpt-5"
API_KEY = os.environ.get("OPENAI_API_KEY", "")

if not API_KEY:
    raise ValueError("OPENAI_API_KEY env var is empty")

SPLITS = ["test"]
MAX_CONCURRENT_GPT = 16
OCR_MAX_DIM = 1500
TEXT_MAX_FONT_SIZE = 48
TEXT_MIN_FONT_SIZE = 10
TEXT_OUTLINE_WIDTH = 2
TEXT_PADDING = 4
MASK_DILATE_MARGIN = 4

os.makedirs(OUT_IMAGES_NO_TEXT, exist_ok=True)
os.makedirs(OUT_IMAGES_TEXT, exist_ok=True)
os.makedirs(OUT_ANNOTATIONS, exist_ok=True)

print("Initializing EasyOCR...")
reader = easyocr.Reader(["en"], gpu=True, quantize=True)
print("Initializing LaMa inpainting model...")
lama = SimpleLama()
print("Initializing AsyncOpenAI client...")
async_client = AsyncOpenAI(api_key=API_KEY)


def get_dominant_brightness(img, x, y, w, h):
    """Average brightness of a rectangular region."""
    img_w, img_h = img.size
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(img_w, x + w)
    y2 = min(img_h, y + h)
    if x2 <= x1 or y2 <= y1:
        return 128
    region = img.crop((x1, y1, x2, y2))
    gray = region.convert("L")
    return np.array(gray).mean()


def wrap_and_fit(
    draw,
    text,
    box_w,
    box_h,
    font_path,
    max_size=TEXT_MAX_FONT_SIZE,
    min_size=TEXT_MIN_FONT_SIZE,
    padding=TEXT_PADDING,
):
    """Largest font size at which wrapped text fits the box. Returns (wrapped_text, font)."""
    usable_w = max(1, box_w - 2 * padding)
    usable_h = max(1, box_h - 2 * padding)

    for size in range(max_size, min_size - 1, -1):
        font = ImageFont.truetype(font_path, size)
        # Cyrillic-width estimate; fall back to Latin if missing.
        avg_char_w = font.getlength("А") or font.getlength("A")
        if avg_char_w <= 0:
            continue

        chars_per_line = max(1, int(usable_w / avg_char_w))
        wrapped = textwrap.fill(text, width=chars_per_line)

        bbox = draw.multiline_textbbox((0, 0), wrapped, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

        if tw <= usable_w and th <= usable_h:
            return wrapped, font

    font = ImageFont.truetype(font_path, min_size)
    avg_char_w = font.getlength("А") or font.getlength("A") or 1
    chars_per_line = max(1, int(usable_w / avg_char_w))
    return textwrap.fill(text, width=chars_per_line), font


def draw_outlined_text(
    draw, xy, text, font, fill, outline_fill, outline_width=TEXT_OUTLINE_WIDTH
):
    """Draw multiline text with an outline stroke."""
    x, y = xy
    for dx in range(-outline_width, outline_width + 1):
        for dy in range(-outline_width, outline_width + 1):
            if dx != 0 or dy != 0:
                draw.multiline_text((x + dx, y + dy), text, font=font, fill=outline_fill)
    draw.multiline_text((x, y), text, font=font, fill=fill)


def render_text_in_box(draw, img, text, box, font_path):
    """Render text into a box: auto-size, wrap, outlined, centered, colour adapts to background."""
    x, y, bw, bh = box["x"], box["y"], box["w"], box["h"]

    brightness = get_dominant_brightness(img, x, y, bw, bh)
    if brightness < 128:
        fill, outline = (255, 255, 255), (0, 0, 0)
    else:
        fill, outline = (0, 0, 0), (255, 255, 255)

    wrapped, font = wrap_and_fit(draw, text, bw, bh, font_path)

    text_bbox = draw.multiline_textbbox((0, 0), wrapped, font=font)
    tw, th = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
    tx = x + (bw - tw) // 2
    ty = y + (bh - th) // 2

    draw_outlined_text(draw, (tx, ty), wrapped, font, fill, outline)


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def dilate_polygon(pts, margin, img_w, img_h):
    """Expand polygon bbox outward by `margin` pixels."""
    x, y, w, h = cv2.boundingRect(pts)
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(img_w, x + w + margin)
    y2 = min(img_h, y + h + margin)
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)


def run_ocr_and_inpaint(sample):
    """OCR + LaMa inpainting on a single sample. Returns boxes + cleaned image."""
    img_path = IMAGE_ROOT / sample["image"]

    if not img_path.exists():
        print(f"  Warning: Image not found: {img_path}")
        return None

    img_cv = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img_cv is None:
        print(f"  Warning: Could not read image: {img_path}")
        return None

    h, w = img_cv.shape[:2]

    # Downscale for faster OCR; coords mapped back via `scale`.
    scale = 1.0
    img_ocr = img_cv
    if max(h, w) > OCR_MAX_DIM:
        scale = OCR_MAX_DIM / max(h, w)
        img_ocr = cv2.resize(img_cv, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    ocr_results = reader.readtext(img_ocr)

    boxes = []
    mask = np.zeros((h, w), dtype=np.uint8)

    for bbox, text, _conf in ocr_results:
        pts = (np.array(bbox, dtype=np.float64) / scale).astype(np.int32)
        x, y, bw, bh = cv2.boundingRect(pts)
        boxes.append({"x": x, "y": y, "w": bw, "h": bh, "text_en": text})
        dilated_pts = dilate_polygon(pts, MASK_DILATE_MARGIN, w, h)
        cv2.fillPoly(mask, [dilated_pts], 255)

    image_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    clean_img = lama(image_pil, Image.fromarray(mask)) if boxes else image_pil

    no_text_path = OUT_IMAGES_NO_TEXT / sample["image"]
    os.makedirs(no_text_path.parent, exist_ok=True)
    clean_img.save(no_text_path)

    return {
        "sample": sample,
        "boxes": boxes,
        "clean_img": clean_img,
    }


async def translate_text_with_gpt_async(boxes, full_text, client, model):
    """Translate per-box OCR text into Ukrainian via GPT."""
    prompt_lines = [f"box {i + 1} - {b['text_en']}" for i, b in enumerate(boxes)]
    prompt_text = "\n".join(prompt_lines)

    prompt = f"""You are translating English-language meme text into Ukrainian for a propaganda detection dataset.

CONTEXT: The text was extracted via OCR from meme images and may contain recognition errors. Use the complete reference text to resolve ambiguities or reconstruct corrupted words.

Reference text (may be cleaner than OCR): {full_text}

OCR text per bounding box:
{prompt_text}

TRANSLATION RULES:
- Translate into natural, colloquial Ukrainian as used in internet memes
- Preserve the original rhetorical intent (sarcasm, fear, outrage, mockery, etc.)
- Keep translations CONCISE — Ukrainian text must fit in the same image space as the English original, so prefer shorter synonyms and phrasings where possible
- Do NOT censor, soften, or editorialize — this is for research purposes and the propaganda effect must be preserved exactly
- If OCR text is garbled, reconstruct the intended English meaning from context before translating

OUTPUT FORMAT (strict):
box 1 - <Ukrainian text>
box 2 - <Ukrainian text>
...

FULL_TEXT: <all boxes combined as a single coherent Ukrainian text>
"""

    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()


def parse_gpt_output(output_text):
    """Extract per-box translations and full text from GPT response."""
    uk_translations = {}

    full_text_uk = (
        output_text.split("FULL_TEXT:", 1)[1].strip()
        if "FULL_TEXT:" in output_text
        else ""
    )

    for line in output_text.splitlines():
        line_stripped = line.strip()
        if line_stripped.lower().startswith("box"):
            parts = line_stripped.split("-", 1)
            if len(parts) == 2:
                uk_translations[parts[0].strip().lower()] = parts[1].strip()

    if not full_text_uk:
        full_text_uk = "\n".join(uk_translations.values())

    return uk_translations, full_text_uk


async def translate_and_draw(intermediate, client, model, font_path, semaphore):
    """Translate one sample and render text into the cleaned image."""
    sample = intermediate["sample"]
    boxes = intermediate["boxes"]
    clean_img = intermediate["clean_img"]

    if boxes:
        async with semaphore:
            gpt_output = await translate_text_with_gpt_async(
                boxes, sample.get("text", ""), client, model
            )
        uk_translations, full_text_uk = parse_gpt_output(gpt_output)
    else:
        uk_translations = {}
        full_text_uk = sample.get("text", "")

    draw = ImageDraw.Draw(clean_img)
    for i, b in enumerate(boxes, start=1):
        text_uk = uk_translations.get(f"box {i}", "")
        if text_uk:
            render_text_in_box(draw, clean_img, text_uk, b, font_path)

    text_path = OUT_IMAGES_TEXT / sample["image"]
    os.makedirs(text_path.parent, exist_ok=True)
    clean_img.save(text_path)

    return {
        "id": sample["id"],
        "labels": sample.get("labels", []),
        "text": full_text_uk,
        "image": sample["image"],
    }


async def process_split(split_name, client, model, font_path):
    """Run the full pipeline on one split."""
    jsonl_path = ANNOTATIONS_DIR / f"{split_name}.jsonl"
    if not jsonl_path.exists():
        print(f"Warning: {jsonl_path} not found, skipping...")
        return

    print(f"\n{'=' * 50}")
    print(f"Processing {split_name} split...")
    print(f"{'=' * 50}")

    data = load_jsonl(jsonl_path)

    # Stage 1: sequential GPU work (OCR + inpainting).
    print(f"[Stage 1] OCR + Inpainting ({len(data)} images)...")
    intermediates = []
    for sample in tqdm(data, desc="OCR+Inpaint"):
        try:
            result = run_ocr_and_inpaint(sample)
            if result:
                intermediates.append(result)
        except Exception as e:
            print(f"  Error in OCR/Inpaint for {sample['id']}: {e}")

    # Stage 2: async GPT translation + text rendering.
    print(f"[Stage 2] Translating {len(intermediates)} samples (max {MAX_CONCURRENT_GPT} concurrent)...")
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_GPT)
    tasks = [
        translate_and_draw(inter, client, model, font_path, semaphore)
        for inter in intermediates
    ]
    translated_data = []
    for coro in tqdm_asyncio.as_completed(tasks, desc="Translate+Draw", total=len(tasks)):
        try:
            result = await coro
            if result:
                translated_data.append(result)
        except Exception as e:
            print(f"  Error in translation: {e}")

    out_jsonl_path = OUT_ANNOTATIONS / f"{split_name}.jsonl"
    save_jsonl(translated_data, out_jsonl_path)
    print(f"Saved {len(translated_data)} annotations to {out_jsonl_path}")


async def main():
    try:
        ImageFont.truetype(FONT_PATH, 28)
        print(f"Using bold font: {FONT_PATH}")
    except Exception as e:
        print(f"ERROR: Could not load font {FONT_PATH}: {e}")
        print("Please install DejaVu fonts or update FONT_PATH.")
        return

    for split in SPLITS:
        await process_split(split, async_client, MODEL, FONT_PATH)

    print(f"\nAll processing complete! Output: {OUT_ROOT}")


if __name__ == "__main__":
    asyncio.run(main())
