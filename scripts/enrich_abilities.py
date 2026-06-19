"""Enrich the ability dataset with diverse, balanced Haki and Physical examples.

The One Piece wiki only has ~4 dedicated Haki pages, so a classifier trained on
them alone learns a weak, low-diversity Haki representation (it gets confused with
Physical Technique). This script pulls the "Haki" and fighting-style sections from
major characters' pages (via the MediaWiki API — Fandom 403s HTML scraping but the
API responds 200), chunks them, caps per character for diversity, and merges them
with the crawler's base dataset to produce a balanced 3-class file.

Usage:
    python scripts/enrich_abilities.py            # reads + overwrites data/abilities.jsonl
    python scripts/enrich_abilities.py --in X --out Y

The input must already contain the crawler's Devil Fruit / Physical Technique /
core-Haki rows (run `scrapy runspider crawler/ability_crawler.py` first).
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://onepiece.fandom.com/api.php"
UA = "Mozilla/5.0 (Onsight educational NLP project)"

# Characters whose pages carry a substantial "Haki" section.
HAKI_CHARACTERS = [
    "Monkey D. Luffy", "Roronoa Zoro", "Sanji", "Jinbe", "Silvers Rayleigh",
    "Shanks", "Dracule Mihawk", "Charlotte Katakuri", "Charlotte Linlin", "Kaidou",
    "Boa Hancock", "Trafalgar D. Water Law", "Yamato", "Sengoku", "Gol D. Roger",
    "Edward Newgate", "Monkey D. Garp", "Portgas D. Ace", "Sabo", "Eustass Kid",
    "Koby", "Marco",
]
# Characters whose pages carry a fighting-style / physical-abilities section.
PHYSICAL_CHARACTERS = [
    "Sanji", "Roronoa Zoro", "Jinbe", "Monkey D. Garp", "Rob Lucci", "Sabo", "Koby",
    "Franky", "Charlotte Katakuri", "Monkey D. Luffy", "Bartholomew Kuma", "Sentomaru",
    "Pedro", "Vista", "Hody Jones", "Gecko Moria", "Kin'emon", "Cavendish",
    "Charlotte Cracker", "King", "Queen",
]
PHYSICAL_SECTION_LINES = {
    "physical abilities", "fighting style", "swordsmanship", "martial arts",
    "hand-to-hand combat", "physical prowess",
}

PER_CHARACTER_CAP = 8
CHUNK_WORDS = 70
MIN_CHUNK_WORDS = 25


def api(params: dict) -> dict:
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    return json.load(urllib.request.urlopen(req, timeout=30))


def wikitext_to_text(wt: str) -> str:
    wt = re.sub(r"(?s)<ref[^>]*>.*?</ref>", " ", wt)
    wt = re.sub(r"(?s)<ref[^>]*/>", " ", wt)
    wt = re.sub(r"(?is)\[\[(?:File|Image):.*?\]\]", " ", wt)        # images first
    for _ in range(3):
        wt = re.sub(r"(?s)\{\{[^{}]*\}\}", " ", wt)                 # templates (nested)
    wt = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", wt)        # links -> label
    wt = re.sub(r"'''?|<[^>]+>", " ", wt)
    wt = re.sub(r"^=+\s*(.*?)\s*=+$", r"\1", wt, flags=re.M)        # headings -> plain
    wt = re.sub(r"^\s*[\*#:;]+", " ", wt, flags=re.M)               # list markers
    wt = re.sub(r"\b(?:thumb|left|right|center|\d+px)\|", "", wt)   # leftover image opts
    wt = re.sub(r"[ \t]{2,}", " ", wt)
    wt = re.sub(r"\n\s*\n+", "\n", wt)
    return wt.strip()


def section_text(name: str, wanted_lines: set[str]) -> str | None:
    """Return cleaned text of the wanted section(s), trying the /Abilities and
    Powers subpage first then the main page."""
    for page in (f"{name}/Abilities and Powers", name):
        try:
            sections = api({"action": "parse", "format": "json", "page": page,
                            "prop": "sections"})["parse"]["sections"]
        except Exception:
            continue
        matches = [s for s in sections if s["line"].strip().lower() in wanted_lines]
        if not matches:
            continue
        parts = []
        for s in matches:
            wt = api({"action": "parse", "format": "json", "page": page,
                      "prop": "wikitext", "section": s["index"]})["parse"]["wikitext"]["*"]
            parts.append(wikitext_to_text(wt))
        return " ".join(parts)
    return None


def chunk(text: str, size: int = CHUNK_WORDS, min_words: int = MIN_CHUNK_WORDS) -> list[str]:
    words = text.split()
    pieces = [" ".join(words[i:i + size]) for i in range(0, len(words), size)]
    return [p for p in pieces if len(p.split()) >= min_words]


def extract(characters: list[str], wanted: set[str], label: str, tag: str,
            sleep: float = 0.15) -> list[dict]:
    rows = []
    for name in characters:
        text = section_text(name, wanted)
        if not text or len(text.split()) < MIN_CHUNK_WORDS:
            continue
        for i, piece in enumerate(chunk(text), start=1):
            rows.append({
                "ability_name": f"{name} {tag} - section {i:02d}",
                "ability_type": label,
                "ability_description": piece,
            })
        time.sleep(sleep)
    return rows


def cap_per_character(rows: list[dict], cap: int, rng: random.Random) -> list[dict]:
    by_char: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        char = re.split(r" (?:Haki|Physical) - section", r["ability_name"])[0]
        by_char[char].append(r)
    out = []
    for group in by_char.values():
        rng.shuffle(group)
        out.extend(group[:cap])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="inp", type=Path, default=Path("data/abilities.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/abilities.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    base = [json.loads(line) for line in args.inp.open(encoding="utf-8")]
    by_type = collections.defaultdict(list)
    for r in base:
        by_type[r["ability_type"]].append(r)

    haki_extra = cap_per_character(
        extract(HAKI_CHARACTERS, {"haki"}, "Haki", "Haki"), PER_CHARACTER_CAP, rng)
    phys_extra = cap_per_character(
        extract(PHYSICAL_CHARACTERS, PHYSICAL_SECTION_LINES, "Physical Technique", "Physical"),
        PER_CHARACTER_CAP, rng)

    combined = (by_type["Devil Fruit"] + by_type["Physical Technique"] + by_type["Haki"]
                + haki_extra + phys_extra)

    seen, final = set(), []
    for r in combined:
        key = (r["ability_type"], r["ability_description"][:200])
        if key in seen:
            continue
        seen.add(key)
        final.append(r)
    rng.shuffle(final)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dist = collections.Counter(r["ability_type"] for r in final)
    print(f"Wrote {len(final)} rows to {args.out}: {dict(dist)}")


if __name__ == "__main__":
    main()
