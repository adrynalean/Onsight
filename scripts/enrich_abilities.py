"""Turn the crawler's base ability dataset into the final training set.

The One Piece wiki only has ~4 dedicated Haki pages, so a classifier trained on
the crawler output alone learns a weak Haki representation and confuses it with
Physical Technique. This script:

  1. Pulls the "Haki" section from major characters' pages via the MediaWiki API
     (Fandom 403s HTML scraping, but the API responds 200), chunks + caps per
     character for diversity, and adds them to the Haki class.
  2. Scrubs Haki-specific vocabulary (haki / busoshoku / kenbunshoku / haoshoku /
     conqueror) out of the Devil Fruit and Physical classes — even dedicated
     fighting-style pages mention Haki, which otherwise poisons "haki" as a signal
     and makes the model dump Haki inputs into Physical Technique.
  3. Balances the three classes to equal size (class weights ~1).

Pipeline (run in order — each step rewrites data/abilities.jsonl):

    scrapy runspider crawler/ability_crawler.py     # base: DF / Physical / core-Haki
    python scripts/enrich_abilities.py              # -> balanced, decontaminated set

The base file is expected to contain the crawler's Devil Fruit, Physical Technique
(dedicated fighting-style pages) and core Haki rows.
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

PER_CHARACTER_CAP = 8
CHUNK_WORDS = 70
MIN_CHUNK_WORDS = 25

# Haki vocabulary scrubbed from the non-Haki classes so "haki" is Haki-exclusive.
HAKI_TERMS = re.compile(r"haki|busoshoku|kenbunshoku|haoshoku|conqueror'?s?", re.IGNORECASE)


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


def haki_section_text(name: str) -> str | None:
    """Cleaned text of a character's Haki section, trying the /Abilities and
    Powers subpage first, then the main page."""
    for page in (f"{name}/Abilities and Powers", name):
        try:
            sections = api({"action": "parse", "format": "json", "page": page,
                            "prop": "sections"})["parse"]["sections"]
        except Exception:
            continue
        matches = [s for s in sections if s["line"].strip().lower() == "haki"]
        if not matches:
            continue
        wt = api({"action": "parse", "format": "json", "page": page,
                  "prop": "wikitext", "section": matches[0]["index"]})["parse"]["wikitext"]["*"]
        return wikitext_to_text(wt)
    return None


def chunk(text: str) -> list[str]:
    words = text.split()
    pieces = [" ".join(words[i:i + CHUNK_WORDS]) for i in range(0, len(words), CHUNK_WORDS)]
    return [p for p in pieces if len(p.split()) >= MIN_CHUNK_WORDS]


def character_haki_rows(rng: random.Random, sleep: float = 0.15) -> list[dict]:
    by_char: dict[str, list[dict]] = collections.defaultdict(list)
    for name in HAKI_CHARACTERS:
        text = haki_section_text(name)
        if not text or len(text.split()) < MIN_CHUNK_WORDS:
            continue
        for i, piece in enumerate(chunk(text), start=1):
            by_char[name].append({
                "ability_name": f"{name} Haki - section {i:02d}",
                "ability_type": "Haki",
                "ability_description": piece,
            })
        time.sleep(sleep)
    rows = []
    for group in by_char.values():
        rng.shuffle(group)
        rows.extend(group[:PER_CHARACTER_CAP])
    return rows


def scrub(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        desc = re.sub(r"\s{2,}", " ", HAKI_TERMS.sub(" ", r["ability_description"])).strip()
        out.append({**r, "ability_description": desc})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="inp", type=Path, default=Path("data/abilities.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/abilities.jsonl"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    base = [json.loads(line) for line in args.inp.open(encoding="utf-8")]
    by_type = collections.defaultdict(list)
    for r in base:
        by_type[r["ability_type"]].append(r)

    devil_fruit = scrub(by_type["Devil Fruit"])
    physical = scrub(by_type["Physical Technique"])
    haki = by_type["Haki"] + character_haki_rows(rng)   # core + diverse character Haki

    for group in (devil_fruit, physical, haki):
        rng.shuffle(group)
    n = min(len(devil_fruit), len(physical), len(haki))
    final = devil_fruit[:n] + physical[:n] + haki[:n]
    rng.shuffle(final)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dist = collections.Counter(r["ability_type"] for r in final)
    print(f"Wrote {len(final)} rows to {args.out}: {dict(dist)} (balanced to {n}/class)")


if __name__ == "__main__":
    main()
