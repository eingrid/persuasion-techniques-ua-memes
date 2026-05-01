# Scrape — UkrMeme Source Collection

Collects Ukrainian war memes from two sources and writes a unified `images_data.json` plus per-source images under `downloaded_images/` (gitignored).

## Files

- `parse.py` — entry point. Two stages:
  1. **Telegram** — Telethon QR-login + photo download from a public channel (`@memcontr` by default).
  2. **Google Drive CSV** — `t.csv` is a third-party crowd-sourced spreadsheet of Ukrainian war memes (one row per meme; the `Google Drive Link` column holds a public Drive file ID). The script samples Ukrainian-language rows and downloads each image by ID.
- `t.csv` — kept in the repository as a record of what was scraped (~5k rows). Columns: `Timestamp, Date Posted, Title, Source Url, Translated Title, Meme Content Type, Country, Language, Meme Template Type, People, Original Filename, Google Drive Link`.

## Usage

```sh
export TELEGRAM_API_ID=...
export TELEGRAM_API_HASH=...

.venv/bin/python scrape/parse.py \
    --telegram 100 \
    --csv 100 \
    --channel memcontr \
    --csv-file scrape/t.csv \
    --output scrape/images_data.json
```

Either source can be disabled by passing `0`. The first Telegram run prompts a QR code; scan it from your account in Telegram → Settings → Devices → Link Desktop Device. Session state is cached in `*.session` files (gitignored).

## Output

- `downloaded_images/` — `tg_<message_id>.jpg` and `csv_<file_id>.jpg`.
- `images_data.json` — list of `{path, text}` records. The Telegram branch leaves `text` as `"Unknown"`; the CSV branch fills it from the spreadsheet `Title` column.

Downstream, the Ukrainian dataset annotators worked off these images; the resulting JSONL lives in `datasets/ukrainian/annotations/test.jsonl`.

## Credentials

`TELEGRAM_API_ID` / `TELEGRAM_API_HASH` come from <https://my.telegram.org>. Rotate any pair that ever appeared in git history.
