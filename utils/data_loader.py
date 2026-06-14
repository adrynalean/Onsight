import re
from pathlib import Path

import pandas as pd

def load_subtitles_dataset(dataset_path):
    dataset_dir = Path(dataset_path).expanduser()
    subtitles_paths = [
        path
        for path in dataset_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {'.ass', '.srt', '.txt'}
    ]

    scripts = []
    episode_num = []
    source_files = []

    for path in subtitles_paths:
        with open(path, 'r', encoding='utf-8-sig', errors='ignore') as file:
            lines = file.readlines()

        if path.suffix.lower() == '.ass':
            # Filter on the 'Dialogue:' prefix directly — robust to header length.
            # (Hardcoding lines[27:] silently dropped dialogue when a file's
            # [Script Info]/[V4+ Styles] header was shorter than 27 lines.)
            lines = [",".join(line.split(',')[9:]) for line in lines if line.startswith('Dialogue:')]
            lines = [line.replace('\\N', ' ').replace('\\n', ' ') for line in lines]
        elif path.suffix.lower() == '.srt':
            # SRT format: strip timestamps, sequence numbers, and blank lines
            lines = [line.strip() for line in lines]
            lines = [line for line in lines if line
                     and not line.isdigit()
                     and '-->' not in line]
        else:
            lines = [line.strip() for line in lines if line.strip()]

        script = " ".join(lines)

        # Robust episode number extraction: grab the last number in the filename.
        numbers = re.findall(r'\d+', path.name)
        episode = int(numbers[-1]) if numbers else 0

        if script.strip():
            scripts.append(script)
            episode_num.append(episode)
            source_files.append(path.name)

    df = pd.DataFrame.from_dict({
        "episode": episode_num,
        "script": scripts,
        "source_file": source_files,
    })
    df = df.sort_values(["episode", "source_file"]).reset_index(drop=True)
    return df
