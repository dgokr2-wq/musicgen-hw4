  !pip install -U -q google-generativeai datasets yt-dlp tqdm
!apt-get update -qq
!apt-get install -y -qq ffmpeg

import os
import subprocess
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

OUTPUT_DIR = "/content/musiccaps_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Загружаем метаданные датасета...")
dataset = load_dataset("google/MusicCaps", split="train")

def download_audio_chunk(row):
    ytid = row["ytid"]
    start_s = row["start_s"]
    end_s = row["end_s"]
    out_path = os.path.join(OUTPUT_DIR, f"{ytid}.wav")

    if os.path.exists(out_path):
        return

    yt_url = f"https://www.youtube.com/watch?v={ytid}"

    try:
        get_url_cmd = ["yt-dlp", "-g", "-f", "bestaudio", yt_url]
        stream_url = subprocess.check_output(
            get_url_cmd,
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()

        ffmpeg_cmd = [
            "ffmpeg",
            "-ss", str(start_s),
            "-to", str(end_s),
            "-i", stream_url,
            "-ar", "32000",
            "-ac", "1",
            "-y",
            out_path
        ]

        subprocess.run(
            ffmpeg_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False
        )

    except Exception:
        pass  

print(f"Начинаем загрузку {len(dataset)} треков. Это займет время...")
with ThreadPoolExecutor(max_workers=8) as executor:
    list(tqdm(
        executor.map(download_audio_chunk, dataset),
        total=len(dataset),
        desc="Скачивание WAV"
    ))

print("Готово.")
