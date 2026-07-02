# One Piece — NLP Series Analysis

An end-to-end NLP / LLM pipeline that analyses the One Piece anime through five modules, tied together in a single Gradio web app.

## 🏴‍☠️ Live demo

**[huggingface.co/spaces/Fluoron/one-piece-analysis](https://huggingface.co/spaces/Fluoron/one-piece-analysis)** — free, always-on HuggingFace Space (CPU). All four interactive modules run in the browser; the Luffy chatbot serves a 4-bit GGUF via `llama-cpp-python`, so no GPU is required.

Trained models on the Hub: [ability classifier](https://huggingface.co/Fluoron/one-piece-ability-classifier) · [Luffy LoRA](https://huggingface.co/Fluoron/one-piece-luffy-chatbot)

> Deployment notes: themes and the character network are precomputed once and committed as stubs (`data/one_piece_s1_*.csv`) so those tabs are instant; the chatbot serves `Llama-3.2-3B-Instruct` (Q4_K_M) on CPU for speed (the fine-tuned 8B LoRA above is the training deliverable). SSR is disabled (`GRADIO_SSR_MODE=False`) so the Space serves on free CPU hardware.

## Modules

| Folder | What it does | Model |
|---|---|---|
| `crawler` | Scrapes ability data (Devil Fruits, Haki, Fighting Styles) from onepiece.fandom.com using Scrapy | — |
| `character_network` | Extracts characters from subtitles with SpaCy NER and renders an interactive PyVis + NetworkX co-occurrence graph | `en_core_web_trf` |
| `text_classification` | Classifies an ability into **Devil Fruit / Haki / Physical Technique** | `microsoft/deberta-v3-small` (fine-tuned) |
| `theme_classifier` | Zero-shot thematic analysis of each episode | `MoritzLaurer/deberta-v3-large-zeroshot-v2` |
| `character_chatbot` | LoRA-fine-tuned Luffy chatbot | `meta-llama/Llama-3.1-8B-Instruct` |
| `semantic_search` | Embeds episode dialogue into overlapping passages and retrieves them by meaning; benchmarked at **95% top-5 passage recall** over 2,036 passages ([results](semantic_search/RESULTS.md)) | `all-MiniLM-L6-v2` + FAISS (hashed-embedding fallback) |

```bash
python -m semantic_search.ingest                       # index episode dialogue
python -m semantic_search.search "who is the pirate hunter?"
python -m semantic_search.evaluate                     # 120-query recall benchmark
```

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_trf
```

Create a `.env` file (see `.env_example`) with your HuggingFace token:

```
huggingface_token=hf_xxxxxxxxxxxxxxxxxxxx
ability_model_path=your-hf-username/one-piece-ability-classifier
luffy_model_path=your-hf-username/one-piece-luffy-chatbot
```

## Data

| Dataset | Purpose | Source |
|---|---|---|
| Subtitles (`.ass` / `.srt`) | Theme + Character network | [one-pace-public-subtitles](https://github.com/one-pace/one-pace-public-subtitles) |
| Dialogue CSV (`one_piece.csv`) | Chatbot training | [HF mirror — `mramazan/...382-777`](https://huggingface.co/datasets/mramazan/One-Piece-Transcripts-with-Character-Names-382-777) (auto-fetched by `build_training_datasets.py`; mirrors [Figshare — Episodes 382–777](https://figshare.com/articles/dataset/One_Piece_Transcripts_with_Character_Names_Episodes_382_777/30188161)) |
| Abilities (`abilities.jsonl`) | Ability classifier training | `scrapy runspider crawler/ability_crawler.py` (base) → `python scripts/enrich_abilities.py` (diverse Haki + decontaminate + balance) |

Expected layout:

```
data/
├── One_Piece_Anime_S1_English/       # anime subtitles, .ass / .srt
├── One_Piece_Anime_S1_English_Text/  # cleaned .txt exports for inspection/debugging
├── ONE_PIECE_S1/                     # optional Netflix live-action subtitles
├── Subtitles/                        # old Naruto reference subtitles
├── one_piece.csv                     # character, text
└── abilities.jsonl                   # ability_name, ability_type, ability_description
```

The Gradio app defaults to `data/One_Piece_Anime_S1_English/`. If OpenSubtitles blocks a download session, rerun the resume script later:

```powershell
.\scripts\download_one_piece_s1_subtitles.ps1
```

To regenerate cleaned text files from downloaded `.ass` / `.srt` subtitles:

```bash
python scripts/convert_subtitles_to_txt.py data/One_Piece_Anime_S1_English data/One_Piece_Anime_S1_English_Text
```

To build the ability dataset (Fandom 403s datacenter IPs, so generate it locally and
commit the result rather than crawling from the training notebook):

```bash
scrapy runspider crawler/ability_crawler.py   # base: Devil Fruit / Physical / core Haki
python scripts/enrich_abilities.py            # + character Haki, scrub Haki terms from
                                              # non-Haki classes, balance to 74/74/74
```

The ability classifier is fine-tuned on this balanced, decontaminated set
(`microsoft/deberta-v3-small`) and pushed to `ability_model_path`. Haki has only ~4
dedicated wiki pages, so `enrich_abilities.py` pulls Haki sections from 22 characters
for diversity and removes Haki vocabulary from the Devil Fruit / Physical classes so
"haki" stays a Haki-only signal.

To build the chatbot datasets:

```bash
python scripts/build_training_datasets.py
```

This creates:

| File | Purpose |
|---|---|
| `data/abilities.jsonl` | Fine-tuning the ability classifier |
| `data/one_piece.csv` | Dialogue/transcript source for the Luffy chatbot |
| `data/luffy_prompts.csv` | Prebuilt prompt examples for inspecting chatbot training data |

## Running

### Crawl abilities
```bash
scrapy runspider crawler/ability_crawler.py
```

### Train models on Kaggle / Colab
Training the ability classifier and Luffy chatbot is done on GPU (Kaggle notebooks recommended). Push the resulting models to your HuggingFace account, then reference them in `gradio_app.py`.

### Launch the web app
```bash
python gradio_app.py
```
