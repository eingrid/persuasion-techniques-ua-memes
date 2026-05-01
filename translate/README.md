# Translate — Propaganda 950 → PTM-UA

Builds the PTM-UA dataset by replacing the English text on every Propaganda 950 meme with a Ukrainian translation while keeping the layout. Three stages per sample:

1. **OCR** the English text with EasyOCR (resized to ≤1500 px).
2. **Inpaint** the original text region with [SimpleLama](https://github.com/enesmsahin/simple-lama-inpainting), producing a clean background.
3. **Translate** each OCR box to natural Ukrainian via GPT-5 (per-box prompt with the full reference text as context), then render the translation back into the original bounding box with a font-size search and an outline for readability.

The script is async — Stage 3 runs up to `MAX_CONCURRENT_GPT` (default 16) GPT calls in parallel; OCR and inpainting are sequential because they share the GPU.

## Usage

```sh
export OPENAI_API_KEY=...

.venv/bin/python translate/translate.py
```

Reads from `datasets/propaganda_950/annotations/<split>.jsonl` (default split: `test`) and writes:

- `datasets/translated/images/<id>.png` — meme with Ukrainian text.
- `datasets/translated/images_no_text/<id>.png` — inpainted, text-stripped intermediate.
- `datasets/translated/annotations/<split>.jsonl` — `{id, labels, text, image}` rows with `text` set to the translated full caption.

## Notes

- Font: hardcoded to `/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf` (DejaVu supports Cyrillic). Override `FONT_PATH` if your system path differs.
- The translation prompt explicitly forbids softening or censoring the original rhetorical intent, since the propaganda effect must be preserved for downstream classification.
