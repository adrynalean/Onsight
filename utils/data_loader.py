from glob import glob
import re
import pandas as pd

def load_subtitles_dataset(dataset_path):
    ass_paths = glob(dataset_path + '/*.ass')
    srt_paths = glob(dataset_path + '/*.srt')
    subtitles_paths = ass_paths + srt_paths

    scripts = []
    episode_num = []

    for path in subtitles_paths:
        with open(path, 'r', encoding='utf-8', errors='ignore') as file:
            lines = file.readlines()

        if path.endswith('.ass'):
            lines = lines[27:]
            lines = [",".join(line.split(',')[9:]) for line in lines]
            lines = [line.replace('\\N', ' ') for line in lines]
        else:
            # SRT format: strip timestamps, sequence numbers, and blank lines
            lines = [line.strip() for line in lines]
            lines = [line for line in lines if line
                     and not line.isdigit()
                     and '-->' not in line]

        script = " ".join(lines)

        # Robust episode number extraction: grab the last number in the filename
        numbers = re.findall(r'\d+', path.split('/')[-1].split('\\')[-1])
        episode = int(numbers[-1]) if numbers else 0

        scripts.append(script)
        episode_num.append(episode)

    df = pd.DataFrame.from_dict({"episode": episode_num, "script": scripts})
    return df
