import os
import re
import csv
import asyncio
import pandas as pd
from telethon import TelegramClient, events
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
env_path = Path(__file__).resolve().parents[1] / '.env'
load_dotenv(dotenv_path=env_path)

# Define API credentials
api_id = os.getenv('TG_API_ID')
api_hash = os.getenv('TG_API_HASH')

# Directory paths
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
MEDIA_DIR = os.path.join(RAW_DIR, "photos")

# Create directories if they don't exist
os.makedirs(MEDIA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Initialize Telegram client
client = TelegramClient('scraping_session', api_id, api_hash)

# List of Telegram channels to scrape
TARGET_CHANNELS = [
    '@qnashcom',
    '@marakibrand',
    '@modernshoppingcenter',
    '@sinayelj',
    '@Leyueqa',
    '@ethio_brand_collection'
]

# CSV file paths
RAW_CSV_PATH = os.path.join(RAW_DIR, "telegram_data.csv")
PROCESSED_CSV_PATH = os.path.join(PROCESSED_DIR, "clean_telegram_data.csv")

def normalize(text):
    """Normalize text by removing extra spaces."""
    return " ".join(text.split())

async def save_message(writer, channel_title, channel_username, message):
    """Save individual message to CSV."""
    media_path = None

    if message.media and hasattr(message.media, 'photo'):
        filename = f"{channel_username.strip('@')}_{message.id}.jpg"
        media_path = os.path.join(MEDIA_DIR, filename)
        await client.download_media(message.media, media_path)

    msg_text = normalize(message.message or "")
    writer.writerow([
        channel_title,
        channel_username,
        message.id,
        msg_text,
        message.date,
        media_path
    ])

async def scrape_channel(channel_username, writer):
    """Scrape messages from a channel."""
    entity = await client.get_entity(channel_username)
    channel_title = entity.title

    async for message in client.iter_messages(entity, limit=1000):
        await save_message(writer, channel_title, channel_username, message)

@client.on(events.NewMessage(chats=TARGET_CHANNELS))
async def realtime_handler(event):
    """Handle new messages in real-time."""
    message = event.message
    media_path = None

    if message.media and hasattr(message.media, 'photo'):
        filename = f"{message.chat.username.strip('@')}_{message.id}.jpg"
        media_path = os.path.join(MEDIA_DIR, filename)
        await client.download_media(message.media, media_path)

    msg_text = normalize(message.message or "")
    with open(RAW_CSV_PATH, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            message.chat.title,
            message.chat.username,
            message.id,
            msg_text,
            message.date,
            media_path
        ])
        print(f"📥 [Real-time] Message from {message.chat.username}")

def preprocess_csv(input_file, output_file):
    """Preprocess the raw CSV file."""
    df = pd.read_csv(input_file)
    df.dropna(subset=["Message"], inplace=True)
    df["Message"] = df["Message"].astype(str).apply(lambda x: normalize(" ".join(x.split())))
    df.to_csv(output_file, index=False)
    print(f"✅ Preprocessed data saved to {output_file}")


async def run_data_collection():
    print("📡 Starting historical message scrape...")

    # 🔑 This connects your client using credentials and creates session
    await client.start()

    # Open CSV to write raw messages
    with open(RAW_CSV_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Channel Title", "Channel Username", "ID", "Message", "Date", "Media Path"])

        for channel in TARGET_CHANNELS:
            await scrape_channel(channel, writer)
            print(f"✅ Finished scraping: {channel}")

    print(f"📄 Raw data saved to {RAW_CSV_PATH}")

    # Preprocess for modeling
    preprocess_csv(RAW_CSV_PATH, PROCESSED_CSV_PATH)
    print("🚀 Listening for new messages (real-time)...")

if __name__ == "__main__":
    asyncio.run(run_data_collection())



