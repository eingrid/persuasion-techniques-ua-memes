import os
import json
import asyncio
import argparse
import requests
import pandas as pd
import qrcode
from telethon import TelegramClient, errors
from telethon.tl.types import MessageMediaPhoto

API_ID = os.environ.get('TELEGRAM_API_ID')
API_HASH = os.environ.get('TELEGRAM_API_HASH')

if not API_ID or not API_HASH:
    raise RuntimeError(
        "TELEGRAM_API_ID and TELEGRAM_API_HASH environment variables must be set. "
        "Obtain credentials at https://my.telegram.org."
    )

API_ID = int(API_ID)

CHANNEL = 'memcontr'
DOWNLOAD_DIR = 'downloaded_images'
CSV_FILE = 't.csv'
OUTPUT_JSON = 'images_data.json'


async def login(client):
    await client.connect()
    
    if await client.is_user_authorized():
        print("Already logged in")
        return
    
    qr_login = await client.qr_login()
    
    qr = qrcode.QRCode(version=1, box_size=2, border=1)
    qr.add_data(qr_login.url)
    qr.make(fit=True)
    qr.print_ascii(invert=True)
    
    print("\nScan with Telegram: Settings > Devices > Link Desktop Device\n")
    
    try:
        await qr_login.wait()
        print("Login successful")
    except errors.SessionPasswordNeededError:
        password = input("Enter 2FA password: ")
        await client.sign_in(password=password)
        print("Login successful")


async def scrape_telegram(client, channel, limit):
    print(f"Scraping {limit} images from Telegram channel: {channel}...")
    
    results = []
    count = 0
    
    async for message in client.iter_messages(channel, limit=None):
        if count >= limit:
            break
        if message.media and isinstance(message.media, MessageMediaPhoto):
            filename = f"{DOWNLOAD_DIR}/tg_{message.id}.jpg"
            await client.download_media(message, filename)
            count += 1
            print(f"[Telegram] Downloaded: {filename}")
            results.append({
                "text": "Unknown",
                "path": filename
            })
    
    print(f"Telegram: {count} images downloaded")
    return results


def download_from_csv(csv_file, limit):
    print(f"Downloading {limit} images from CSV: {csv_file}...")
    
    df = pd.read_csv(csv_file)
    df = df[df['Language'] == 'Ukrainian']
    df = df.sample(frac=1)
    df['img_id'] = df['Google Drive Link'].apply(lambda x: x.split("id=")[-1])
    
    results = []
    count = 0
    
    for idx, row in df.iterrows():
        if count >= limit:
            break
        
        file_id = row['img_id']
        save_path = f"{DOWNLOAD_DIR}/csv_{file_id}.jpg"
        
        if os.path.exists(save_path):
            print(f"[CSV] Already exists: {save_path}")
            results.append({
                "text": str(row.get('Title', 'Unknown')),
                "path": save_path
            })
            count += 1
            continue
        
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        response = requests.get(url, stream=True)
        
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print(f"[CSV] Downloaded: {save_path}")
            results.append({
                "text": str(row.get('Title', 'Unknown')),
                "path": save_path
            })
            count += 1
        else:
            print(f"[CSV] Failed: {file_id} (status {response.status_code})")
    
    print(f"CSV: {count} images downloaded")
    return results


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--telegram', type=int, default=100, help='Number of images from Telegram')
    parser.add_argument('--csv', type=int, default=100, help='Number of images from CSV')
    parser.add_argument('--channel', type=str, default=CHANNEL, help='Telegram channel username')
    parser.add_argument('--csv-file', type=str, default=CSV_FILE, help='Path to CSV file')
    parser.add_argument('--output', type=str, default=OUTPUT_JSON, help='Output JSON file')
    args = parser.parse_args()
    
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    
    all_results = []
    
    if args.telegram > 0:
        client = TelegramClient('session', API_ID, API_HASH)
        await login(client)
        tg_results = await scrape_telegram(client, args.channel, args.telegram)
        all_results.extend(tg_results)
        await client.disconnect()
    
    if args.csv > 0:
        csv_results = download_from_csv(args.csv_file, args.csv)
        all_results.extend(csv_results)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved {len(all_results)} entries to {args.output}")


if __name__ == '__main__':
    asyncio.run(main())