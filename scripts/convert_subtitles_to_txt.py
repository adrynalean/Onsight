from __future__ import annotations

import argparse
import re
from pathlib import Path


TIMESTAMP_RE = re.compile(r"^\d{1,2}:\d{2}:\d{2}[,.]\d{2,3}\s+-->\s+")
ASS_TAG_RE = re.compile(r"\{[^}]*\}")
HTML_TAG_RE = re.compile(r"<[^>]+>")


def clean_line(line: str) -> str:
    line = line.strip().replace(r"\N", " ").replace(r"\n", " ")
    line = ASS_TAG_RE.sub("", line)
    line = HTML_TAG_RE.sub("", line)
    line = re.sub(r"\s+", " ", line)
    return line.strip()


def read_ass(path: Path) -> list[str]:
    lines: list[str] = []
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not raw_line.startswith("Dialogue:"):
            continue

        parts = raw_line.split(",", 9)
        if len(parts) < 10:
            continue

        line = clean_line(parts[9])
        if line:
            lines.append(line)
    return lines


def read_srt(path: Path) -> list[str]:
    lines: list[str] = []
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.isdigit() or TIMESTAMP_RE.match(line):
            continue

        line = clean_line(line)
        if line:
            lines.append(line)
    return lines


def convert_file(path: Path, output_dir: Path) -> Path:
    if path.suffix.lower() == ".ass":
        lines = read_ass(path)
    elif path.suffix.lower() == ".srt":
        lines = read_srt(path)
    else:
        raise ValueError(f"Unsupported subtitle extension: {path.suffix}")

    output_path = output_dir / f"{path.stem}.txt"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert subtitle files to plain text.")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(
        path
        for path in args.input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".ass", ".srt"}
    )

    for path in paths:
        output_path = convert_file(path, args.output_dir)
        print(f"{path.name} -> {output_path.name}")

    print(f"Converted {len(paths)} subtitle files.")


if __name__ == "__main__":
    main()
