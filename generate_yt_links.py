#!/usr/bin/env python3
"""
Append a YouTube link column to 'musicsheets.csv' using pandas, with known columns:
    - title
    - genre
    - artists  (may contain multiple artists separated by commas)

This version uses only yt-dlp for search (no API key required).

Run with:
    python generate_youtube_links_pandas_ytdlp.py

Requirements:
    pip install pandas yt-dlp
"""

import os
import time
import sys
import pandas as pd
from typing import Optional

LINK_COL_NAME = "youtube_url"
SLEEP_SEC = 0.1  # Reduced from 0.2 to 0.1 for faster processing
PREFER_OFFICIAL_AUDIO = True

TITLE_COL = "title"
ARTISTS_COL = "artists"

def find_input_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "musicsheets.csv"),  # Look in current directory first
        "/Users/bowenxia/musicsheets.csv"  # Fallback to absolute path
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Could not find musicsheets.csv")

def derive_output_path(input_path: str) -> str:
    base, ext = os.path.splitext(input_path)
    return f"{base}_with_youtube{ext or '.csv'}"

def build_query(title: str, artists: str) -> str:
    # Use first artist if multiple provided
    main_artist = artists.split(",")[0].strip() if artists else ""
    parts = [str(title).strip()]
    if main_artist:
        parts.append(main_artist)
    if PREFER_OFFICIAL_AUDIO:
        parts.append("official audio")
    return " ".join([p for p in parts if p])

def search_ytdlp(query: str) -> Optional[str]:
    try:
        import yt_dlp  # type: ignore
    except Exception:
        print("[error] yt-dlp is not installed. Please run: pip install yt-dlp")
        return None
    
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "extract_flat": True,
        "default_search": "ytsearch",
        "socket_timeout": 10,  # 10 second timeout
        "retries": 2,  # Retry failed requests
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"ytsearch1:{query}", download=False)
            entries = info.get("entries", [])
            if not entries:
                return None
            url = entries[0].get("url")
            if url and url.startswith("http"):
                return url
            vid = entries[0].get("id")
            if vid:
                return f"https://www.youtube.com/watch?v={vid}"
            return None
    except Exception as e:
        print(f"[warning] Failed to search for '{query}': {str(e)[:100]}...")
        return None

def main():
    input_path = find_input_path()
    output_path = derive_output_path(input_path)

    print(f"[info] Reading: {input_path}")
    df = pd.read_csv(input_path)
    print(f"[info] Loaded {len(df)} rows from CSV")

    if TITLE_COL not in df.columns or ARTISTS_COL not in df.columns:
        raise ValueError(f"Expected columns '{TITLE_COL}' and '{ARTISTS_COL}' not found in CSV. Found: {df.columns.tolist()}")

    print(f"[info] Starting YouTube search for {len(df)} songs...")
    print(f"[info] Estimated time: {len(df) * SLEEP_SEC:.1f} seconds (sleep delays only)")
    
    def gen_link(row) -> str:
        query = build_query(row[TITLE_COL], row[ARTISTS_COL])
        link = search_ytdlp(query)
        if SLEEP_SEC > 0:
            time.sleep(SLEEP_SEC)
        return link or ""

    # Add progress tracking
    total_rows = len(df)
    for idx, row in df.iterrows():
        if idx % 10 == 0:  # Show progress every 10 rows
            print(f"[progress] Processing row {idx + 1}/{total_rows} ({(idx + 1) / total_rows * 100:.1f}%)")
        
        query = build_query(row[TITLE_COL], row[ARTISTS_COL])
        link = search_ytdlp(query)
        df.at[idx, LINK_COL_NAME] = link or ""
        
        if SLEEP_SEC > 0:
            time.sleep(SLEEP_SEC)

    print(f"[info] Writing: {output_path}")
    df.to_csv(output_path, index=False)
    print(f"[done] Rows written: {len(df)}")
    print(f"[done] New column: {LINK_COL_NAME}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)
