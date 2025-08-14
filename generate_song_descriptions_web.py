import csv
import os
import time
from typing import Dict, Any
from openai import OpenAI, APIError, RateLimitError, APITimeoutError
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ---------- CONFIG ----------
INPUT_CSV  = "musicsheets.csv"      # your input CSV
OUTPUT_CSV = "musicsheets_described.csv"     # output file
TITLE_COL  = "Title"                   # adjust to your sheet
ARTIST_COL = "Artist"                  # adjust to your sheet
ALBUM_COL  = "album"                   # optional
YEAR_COL   = "year"                    # optional
MODEL      = "gpt-4o-mini"             # cost-effective, good quality

MAX_RETRIES = 5
BACKOFF_S   = 3

# ---------- SYSTEM PROMPT ----------
SYSTEM_PROMPT = """You write compact, specific music descriptions optimized for semantic search.
Rules:
- Output length 300–500 characters.
- Do NOT include title or artist name in the output.
- Include objective features when confidently known (tempo BPM, musical key, time signature, duration, instrumentation, structure).
- Use rich, concrete keywords (e.g., warm guitar, airy falsetto, programmed drums, sub-bass, lo-fi sheen).
- Capture themes/mood without quoting lyrics; never output verbatim lyrics.
- Prefer facts found via web search; if not confidently known, omit rather than guess.
- Single line output; no headers, no markdown, no extra explanations.
"""

# ---------- USER PROMPT BUILDER ----------
def build_user_prompt(row: Dict[str, Any]) -> str:
    title  = str(row.get(TITLE_COL, "") or "").strip()
    artist = str(row.get(ARTIST_COL, "") or "").strip()
    album  = str(row.get(ALBUM_COL, "") or "").strip()
    year   = str(row.get(YEAR_COL, "") or "").strip()

    # You can add more fields (genre, bpm, key) if your sheet has them.
    # We pass title/artist as context ONLY; the model must NOT echo them.
    lines = [
        f"Song title: {title}" if title else "",
        f"Artist: {artist}" if artist else "",
        f"Album: {album}" if album else "",
        f"Year: {year}" if year else "",
        "Task: Produce a 300–500 character, keyword-dense description with objective musical features when confidently known. No lyric quotes. Do not include the title or artist name in the output."
    ]
    return "\n".join([l for l in lines if l])

# ---------- MODEL CALL (web search enabled) ----------
def call_model(client: OpenAI, user_prompt: str) -> str:
    tools = [{"type": "web_search"}]  # enable web browsing for grounding

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.responses.create(
                model=MODEL,
                system=SYSTEM_PROMPT,
                input=user_prompt,
                tools=tools,                 # <-- key line: turns on web search
                temperature=0.5,
                max_output_tokens=320        # enough for ~500 chars
            )
            return (resp.output_text or "").strip()
        except (RateLimitError, APITimeoutError):
            if attempt == MAX_RETRIES:
                raise
            time.sleep(BACKOFF_S * attempt)
        except APIError:
            raise

# ---------- MAIN ----------
def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY in your environment.")

    client = OpenAI(api_key=api_key)

    with open(INPUT_CSV, newline="", encoding="utf-8") as f_in, \
         open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f_out:

        reader = csv.DictReader(f_in)
        if not reader.fieldnames:
            raise RuntimeError("Input CSV has no header.")
        fieldnames = list(reader.fieldnames)
        out_col = "semantic_description"
        if out_col not in fieldnames:
            fieldnames.append(out_col)

        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            prompt = build_user_prompt(row)
            try:
                desc = call_model(client, prompt)
            except Exception as e:
                desc = ""  # or put str(e) if you want to debug in output
            row[out_col] = desc
            writer.writerow(row)

    print(f"Done. Wrote {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
