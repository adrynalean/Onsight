---
title: One Piece Analysis
emoji: 🏴‍☠️
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: 5.23.3
app_file: gradio_app.py
pinned: false
license: mit
short_description: One Piece NLP dashboard — themes, network, abilities, Luffy
---

# One Piece — NLP Series Analysis

An end-to-end NLP / LLM dashboard for the One Piece anime, with four modules:

- **Theme classification** — zero-shot scoring of episode themes.
- **Character network** — SpaCy NER + an interactive PyVis co-occurrence graph.
- **Ability classifier** — fine-tuned DeBERTa-v3 sorting an ability into Devil Fruit / Haki / Physical Technique. Model: [`Fluoron/one-piece-ability-classifier`](https://huggingface.co/Fluoron/one-piece-ability-classifier).
- **Luffy chatbot** — LoRA-fine-tuned Llama 3.1 8B, served on CPU as a 4-bit GGUF. Model: [`Fluoron/one-piece-luffy-chatbot`](https://huggingface.co/Fluoron/one-piece-luffy-chatbot).

Runs free on CPU. Code: [github.com/adrynalean/Onsight](https://github.com/adrynalean/Onsight).
