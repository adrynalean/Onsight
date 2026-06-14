from __future__ import annotations

import argparse
import csv
import html
import json
import re
import time
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests


TRANSCRIPT_URL = (
    "https://huggingface.co/datasets/"
    "mramazan/One-Piece-Transcripts-with-Character-Names-382-777/"
    "resolve/main/onepiece.csv"
)

FANDOM_API = "https://onepiece.fandom.com/api.php"
ABILITY_CATEGORIES = {
    "Category:Devil_Fruits": "Devil Fruit",
    "Category:Fighting_Styles": "Physical Technique",
}
HAKI_PAGES = [
    "Haki",
    "Haki/Armament Haki",
    "Haki/Observation Haki",
    "Haki/Supreme King Haki",
]


def request_json(session: requests.Session, params: dict) -> dict:
    response = session.get(FANDOM_API, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def category_pages(
    session: requests.Session,
    category: str,
    seen_categories: set[str] | None = None,
) -> Iterable[dict]:
    if seen_categories is None:
        seen_categories = set()
    if category in seen_categories:
        return
    seen_categories.add(category)

    params = {
        "action": "query",
        "format": "json",
        "list": "categorymembers",
        "cmtitle": category,
        "cmtype": "page|subcat",
        "cmlimit": "500",
    }

    while True:
        data = request_json(session, params)
        for member in data["query"]["categorymembers"]:
            if member.get("ns") == 14:
                yield from category_pages(session, member["title"], seen_categories)
            else:
                yield member
        if "continue" not in data:
            break
        params.update(data["continue"])


def page_info(session: requests.Session, title: str) -> dict | None:
    data = request_json(
        session,
        {
            "action": "query",
            "format": "json",
            "titles": title,
        },
    )
    page = next(iter(data["query"]["pages"].values()))
    if "missing" in page:
        return None
    return page


def html_to_text(raw_html: str) -> str:
    raw_html = re.sub(r"(?is)<(script|style|table|sup|nav|aside).*?</\1>", " ", raw_html)
    raw_html = re.sub(r"(?is)<br\s*/?>", "\n", raw_html)
    raw_html = re.sub(r"(?is)</p>|</li>|</h[1-6]>", "\n", raw_html)
    text = re.sub(r"(?is)<[^>]+>", " ", raw_html)
    text = html.unescape(text)
    text = re.sub(r"\[[^\]]*edit[^\]]*\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n\s+", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def page_extract(session: requests.Session, pageid: int) -> str:
    data = request_json(
        session,
        {
            "action": "parse",
            "format": "json",
            "prop": "text",
            "pageid": str(pageid),
        },
    )
    raw_html = data["parse"]["text"]["*"]
    extract = html_to_text(raw_html)
    extract = re.split(
        r"\n(?:Trivia|References|External Links|Site Navigation|Navigation)\n",
        extract,
        flags=re.IGNORECASE,
    )[0]
    extract = re.sub(r"\n{3,}", "\n\n", extract)
    return extract.strip()


def clean_ability_title(title: str) -> bool:
    ignored_prefixes = (
        "Category:",
        "List of",
        "Template:",
        "User:",
        "Forum:",
        "One Piece Wiki:",
    )
    return bool(title.strip()) and not title.startswith(ignored_prefixes)


def split_text_chunks(text: str, min_words: int = 35, max_words: int = 180) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for paragraph in re.split(r"\n{1,}", text):
        paragraph = paragraph.strip()
        words = paragraph.split()
        if len(words) < 8:
            continue

        if current and current_words + len(words) > max_words:
            if current_words >= min_words:
                chunks.append(" ".join(current))
            current = []
            current_words = 0

        current.append(paragraph)
        current_words += len(words)

    if current_words >= min_words:
        chunks.append(" ".join(current))

    return chunks


def build_abilities(output_path: Path, sleep_seconds: float = 0.15) -> pd.DataFrame:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Onsight dataset builder "
                "(educational NLP project; contact via GitHub adrynalean/Onsight)"
            )
        }
    )

    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for category, simplified_type in ABILITY_CATEGORIES.items():
        for page in category_pages(session, category):
            title = page["title"].strip()
            if not clean_ability_title(title):
                continue

            key = (title, simplified_type)
            if key in seen:
                continue

            description = page_extract(session, page["pageid"])
            if len(description.split()) < 12:
                continue

            rows.append(
                {
                    "ability_name": title,
                    "ability_type": simplified_type,
                    "ability_description": description,
                }
            )
            seen.add(key)
            time.sleep(sleep_seconds)

    for title in HAKI_PAGES:
        page = page_info(session, title)
        if not page:
            continue

        description = page_extract(session, page["pageid"])
        chunks = split_text_chunks(description)
        for index, chunk in enumerate(chunks, start=1):
            rows.append(
                {
                    "ability_name": f"{title} - section {index:02d}",
                    "ability_type": "Haki",
                    "ability_description": chunk,
                }
            )
        time.sleep(sleep_seconds)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    return pd.DataFrame(rows)


def download_transcripts(output_path: Path) -> pd.DataFrame:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(TRANSCRIPT_URL, timeout=120)
    response.raise_for_status()
    output_path.write_bytes(response.content)

    df = pd.read_csv(output_path)
    expected = {"episode", "start", "end", "character", "text"}
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Transcript CSV is missing columns: {sorted(missing)}")
    return df


def build_luffy_prompts(transcript_path: Path, output_path: Path) -> pd.DataFrame:
    transcript_df = pd.read_csv(transcript_path)
    transcript_df = transcript_df.dropna(subset=["character", "text"]).reset_index(drop=True)
    transcript_df["text"] = transcript_df["text"].astype(str).str.replace(
        r"\(.*?\)", "", regex=True
    ).str.strip()
    transcript_df["number_of_words"] = transcript_df["text"].str.split().apply(len)

    # Exact 'Luffy' only — the dataset labels solo Luffy lines as "Luffy" and
    # has decoy/group labels ("Not Luffy", "Luffy & Usopp") that must NOT match.
    # Kept consistent with character_chatbot.CharacterChatBot.load_data.
    transcript_df["luffy_response_flag"] = (
        (transcript_df["character"] == "Luffy")
        & (transcript_df["number_of_words"] > 5)
    )

    system_prompt = (
        'You are Monkey D. Luffy from the anime "One Piece". '
        "Respond exactly as Luffy would: carefree, enthusiastic, and direct. "
        "You dream of becoming King of the Pirates. You care deeply about your crew and friends. "
        "You are fearless, a little oblivious to complex things, and always hungry.\n"
    )

    prompts: list[str] = []
    for index in transcript_df[transcript_df["luffy_response_flag"]].index:
        if index == 0:
            continue
        previous_line = transcript_df.loc[index - 1, "text"]
        response_line = transcript_df.loc[index, "text"]
        if previous_line and response_line:
            prompts.append(f"{system_prompt}{previous_line}\n{response_line}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    prompts_df = pd.DataFrame({"prompt": prompts})
    prompts_df.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)
    return prompts_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build One Piece training datasets.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--skip-abilities", action="store_true")
    parser.add_argument("--skip-transcripts", action="store_true")
    args = parser.parse_args()

    data_dir = args.data_dir
    ability_path = data_dir / "abilities.jsonl"
    transcript_path = data_dir / "one_piece.csv"
    prompt_path = data_dir / "luffy_prompts.csv"

    if not args.skip_transcripts:
        transcripts = download_transcripts(transcript_path)
        prompts = build_luffy_prompts(transcript_path, prompt_path)
        print(
            f"Transcript rows: {len(transcripts):,}; "
            f"Luffy prompt rows: {len(prompts):,}; saved to {transcript_path} and {prompt_path}"
        )

    if not args.skip_abilities:
        abilities = build_abilities(ability_path)
        counts = abilities["ability_type"].value_counts().to_dict()
        print(f"Ability rows: {len(abilities):,}; class counts: {counts}; saved to {ability_path}")


if __name__ == "__main__":
    main()
