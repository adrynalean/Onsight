import os
# Force Gradio SSR off BEFORE importing gradio. HF Spaces sets GRADIO_SSR_MODE=1,
# and SSR (needs Node) hangs a CPU Space on "Starting" with a non-interactive UI.
# Setting it here overrides HF's value regardless of whether launch(ssr_mode=...)
# wins, so the queue/backend actually serves.
os.environ["GRADIO_SSR_MODE"] = "False"

import gradio as gr
from dotenv import load_dotenv
load_dotenv()

# Heavy module classes are imported lazily inside each handler so the app can
# start on a CPU-only Space even if a given module's deps aren't installed — a
# feature only fails (clearly) when its tab is actually used.

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SUBTITLES_PATH = os.path.join(PROJECT_ROOT, "data", "One_Piece_Anime_S1_English")
DEFAULT_TEXT_PATH = os.path.join(PROJECT_ROOT, "data", "One_Piece_Anime_S1_English_Text")
DEFAULT_THEME_SAVE_PATH = os.path.join(PROJECT_ROOT, "data", "one_piece_s1_theme_scores.csv")
DEFAULT_NER_SAVE_PATH = os.path.join(PROJECT_ROOT, "data", "one_piece_s1_ners.csv")
DEFAULT_ABILITIES_PATH = os.path.join(PROJECT_ROOT, "data", "abilities.jsonl")
DEFAULT_TRANSCRIPT_PATH = os.path.join(PROJECT_ROOT, "data", "one_piece.csv")
DEFAULT_THEMES = "freedom, adventure, friendship, dreams, justice, sacrifice, loyalty, betrayal, family, courage"
DEFAULT_ABILITY_MODEL_PATH = os.getenv("ability_model_path", "Fluoron/one-piece-ability-classifier")
DEFAULT_LUFFY_MODEL_PATH = os.getenv("luffy_model_path", "Fluoron/one-piece-luffy-chatbot")
# CPU chatbot: 4-bit GGUF served by llama.cpp (no GPU needed). Uses the public,
# ungated Llama 3.1 8B GGUF + the Luffy system prompt below for a free, robust
# live demo. The fine-tuned LoRA lives at Fluoron/one-piece-luffy-chatbot; point
# luffy_gguf_repo/file at a converted GGUF of it to serve the fine-tune instead.
DEFAULT_LUFFY_GGUF_REPO = os.getenv("luffy_gguf_repo", "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF")
DEFAULT_LUFFY_GGUF_FILE = os.getenv("luffy_gguf_file", "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")

LUFFY_SYSTEM_PROMPT = (
    'You are Monkey D. Luffy from the anime "One Piece". '
    "Respond exactly as Luffy would: carefree, enthusiastic, and direct. "
    "You dream of becoming King of the Pirates. You care deeply about your crew and friends. "
    "You are fearless, a little oblivious to complex things, and always hungry. "
    "Keep responses short and energetic, true to Luffy's speech patterns."
)
_LUFFY_LLM = None

def get_themes(theme_list_str, subtitles_path, save_path):
    from theme_classifier import ThemeClassifier
    theme_list = [theme.strip() for theme in theme_list_str.split(',') if theme.strip()]
    subtitles_path = subtitles_path.strip() or DEFAULT_SUBTITLES_PATH
    save_path = save_path.strip() or None
    theme_classifier = ThemeClassifier(theme_list)
    output_df = theme_classifier.get_themes(subtitles_path, save_path)

    # Remove 'dialogue' from themes before plotting
    theme_list = [theme for theme in theme_list if theme != 'dialogue']
    output_df = output_df[theme_list].sum().reset_index()
    output_df.columns = ['Theme', 'Score']

    output_chart = gr.BarPlot(
        output_df,
        x="Theme",
        y="Score",
        title="One Piece — Series Themes",
        tooltip=["Theme", "Score"],
        vertical=False,
        width=500,
        height=260
    )

    return output_chart

def get_character_network(subtitles_path, ner_path):
    from character_network import NamedEntityRecognizer, CharacterNetworkGenerator
    subtitles_path = subtitles_path.strip() or DEFAULT_SUBTITLES_PATH
    ner_path = ner_path.strip() or None
    ner = NamedEntityRecognizer()
    ner_df = ner.get_ners(subtitles_path, ner_path)

    character_network_generator = CharacterNetworkGenerator()
    relationship_df = character_network_generator.generate_character_network(ner_df)
    html = character_network_generator.draw_network_graph(relationship_df)

    return html

def classify_text(text_classification_model, text_classification_data_path, text_to_classify):
    from text_classification import AbilityClassifier
    model_path = text_classification_model.strip()
    if not model_path:
        raise ValueError(
            "Enter your Hugging Face ability model path, e.g. "
            "your-username/one-piece-ability-classifier. If it does not exist yet, "
            "the classifier will train from data/abilities.jsonl and push it."
        )

    text_classification_data_path = text_classification_data_path.strip() or DEFAULT_ABILITIES_PATH
    ability_classifier = AbilityClassifier(
        model_path=model_path,
        data_path=text_classification_data_path,
        huggingface_token=os.getenv('huggingface_token')
    )

    output = ability_classifier.classify_ability(text_to_classify)
    output = output[0]

    return output

def _load_luffy_llm():
    # Lazy, cached load of the 4-bit GGUF so the Space boots instantly and only
    # pulls the ~4.5GB model into RAM the first time someone chats.
    global _LUFFY_LLM
    if _LUFFY_LLM is None:
        from llama_cpp import Llama
        _LUFFY_LLM = Llama.from_pretrained(
            repo_id=DEFAULT_LUFFY_GGUF_REPO,
            filename=DEFAULT_LUFFY_GGUF_FILE,
            n_ctx=2048,
            n_threads=os.cpu_count(),
            verbose=False,
        )
    return _LUFFY_LLM


def chat_with_character_chatbot(message, history):
    llm = _load_luffy_llm()
    messages = [{"role": "system", "content": LUFFY_SYSTEM_PROMPT}]
    for entry in history:
        if isinstance(entry, dict):
            messages.append({"role": entry["role"], "content": entry["content"]})
        else:
            messages.append({"role": "user", "content": entry[0]})
            messages.append({"role": "assistant", "content": entry[1]})
    messages.append({"role": "user", "content": message})

    result = llm.create_chat_completion(
        messages=messages, max_tokens=200, temperature=0.7, top_p=0.9,
    )
    return result["choices"][0]["message"]["content"].strip()


# "Grand Line" theme — ocean blue + straw-hat gold.
OP_THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.amber,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Poppins"), "ui-sans-serif", "sans-serif"],
)

OP_CSS = """
.gradio-container { background: #eef4f8 !important; }
#op-header {
    background: linear-gradient(135deg, #0a2540 0%, #14406b 55%, #1b6ca8 100%);
    border: 1px solid rgba(244,196,48,0.45); border-radius: 16px;
    padding: 30px 34px; margin-bottom: 6px; box-shadow: 0 10px 28px rgba(10,37,64,0.28);
}
#op-header h1 { color:#fff; margin:0; font-size:2.3rem; letter-spacing:3px; font-weight:700; }
#op-header h1 .accent { color:#f4c430; }
#op-header p { color:#cfe3f2; margin:8px 0 0; font-size:1.02rem; }
.op-section {
    background: linear-gradient(90deg,#0a2540,#14406b); border-left: 6px solid #f4c430;
    border-radius: 10px; padding: 13px 20px; margin: 22px 0 6px; color:#fff;
}
.op-section .t { font-size:1.3rem; font-weight:600; }
.op-section .s { color:#cfe3f2; font-size:0.88rem; }
button.op-btn { background:#f4c430 !important; color:#0a2540 !important; font-weight:600 !important; border:none !important; }
button.op-btn:hover { background:#ffd23f !important; }
#op-footer { text-align:center; color:#5b7a93; font-size:0.85rem; padding:22px 0 8px; }
"""


def _section(icon, title, subtitle):
    return gr.HTML(
        f'<div class="op-section"><span class="t">{icon} {title}</span>'
        f'<div class="s">{subtitle}</div></div>'
    )


def main():
    with gr.Blocks(theme=OP_THEME, css=OP_CSS, title="One Piece — Series Analysis") as iface:
        gr.HTML(
            '<div id="op-header">'
            '<h1>🏴‍☠️ ONE <span class="accent">PIECE</span> &nbsp;·&nbsp; Series Analysis</h1>'
            '<p>AI / NLP on the Grand Line — story themes, crew bonds, ability types, '
            'and a chat with Luffy</p>'
            '</div>'
        )

        # Theme Classification
        _section("🌊", "Themes of the Grand Line",
                 "Zero-shot classification of recurring themes across the subtitles")
        with gr.Row():
            with gr.Column():
                plot = gr.BarPlot()
            with gr.Column():
                theme_list = gr.Textbox(label="Themes", value=DEFAULT_THEMES)
                theme_subtitles_path = gr.Textbox(label="Subtitles or script path", value=DEFAULT_SUBTITLES_PATH)
                save_path = gr.Textbox(label="Save path", value=DEFAULT_THEME_SAVE_PATH)
                get_themes_button = gr.Button("Get themes", elem_classes="op-btn")
                get_themes_button.click(
                    get_themes,
                    inputs=[theme_list, theme_subtitles_path, save_path],
                    outputs=[plot],
                )

        # Character Network
        _section("🤝", "The crew &amp; their bonds",
                 "Named-entity recognition + a co-occurrence graph of characters")
        with gr.Row():
            with gr.Column():
                network_html = gr.HTML()
            with gr.Column():
                network_subtitles_path = gr.Textbox(label="Subtitles or script path", value=DEFAULT_SUBTITLES_PATH)
                ner_path = gr.Textbox(label="NER save path", value=DEFAULT_NER_SAVE_PATH)
                get_network_graph_button = gr.Button("Get character network", elem_classes="op-btn")
                get_network_graph_button.click(
                    get_character_network,
                    inputs=[network_subtitles_path, ner_path],
                    outputs=[network_html],
                )

        # Ability Classification
        _section("⚔️", "Devil Fruit · Haki · Physical",
                 "Classify any ability description into one of the three combat types")
        with gr.Row():
            with gr.Column():
                text_classification_output = gr.Textbox(label="Predicted ability type")
            with gr.Column():
                text_classification_model = gr.Textbox(label="Model path", value=DEFAULT_ABILITY_MODEL_PATH)
                text_classification_data_path = gr.Textbox(label="Data path", value=DEFAULT_ABILITIES_PATH)
                text_to_classify = gr.Textbox(
                    label="Ability description", lines=3,
                    placeholder="e.g. The Gomu Gomu no Mi turns the user's body into rubber...",
                )
                classify_text_button = gr.Button("Classify ability", elem_classes="op-btn")
                classify_text_button.click(
                    classify_text,
                    inputs=[text_classification_model, text_classification_data_path, text_to_classify],
                    outputs=[text_classification_output],
                )

        # Luffy Chatbot
        _section("🎙️", "Chat with Luffy",
                 "A fine-tuned Llama 3.1 8B that talks like the future Pirate King")
        gr.ChatInterface(chat_with_character_chatbot, type="messages")

        gr.HTML('<div id="op-footer">Built with 🤗 Transformers · SpaCy · PyVis · Gradio'
                ' &nbsp;|&nbsp; One Piece NLP analysis suite</div>')

    # ssr_mode=False: Gradio 5's experimental SSR layer trips HuggingFace Spaces'
    # health check and leaves the Space stuck on "Starting".
    iface.launch(ssr_mode=False)


if __name__ == '__main__':
    main()
