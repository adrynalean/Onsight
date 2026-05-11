import gradio as gr
from theme_classifier import ThemeClassifier
from character_network import NamedEntityRecognizer, CharacterNetworkGenerator
from text_classification import AbilityClassifier
from character_chatbot import CharacterChatBot
import os
from dotenv import load_dotenv
load_dotenv()

def get_themes(theme_list_str, subtitles_path, save_path):
    theme_list = theme_list_str.split(',')
    theme_classifier = ThemeClassifier(theme_list)
    output_df = theme_classifier.get_themes(subtitles_path, save_path)

    # Remove 'dialogue' from themes before plotting
    theme_list = [theme for theme in theme_list if theme != 'dialogue']
    output_df = output_df[theme_list]

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
    ner = NamedEntityRecognizer()
    ner_df = ner.get_ners(subtitles_path, ner_path)

    character_network_generator = CharacterNetworkGenerator()
    relationship_df = character_network_generator.generate_character_network(ner_df)
    html = character_network_generator.draw_network_graph(relationship_df)

    return html

def classify_text(text_classification_model, text_classification_data_path, text_to_classify):
    ability_classifier = AbilityClassifier(
        model_path=text_classification_model,
        data_path=text_classification_data_path,
        huggingface_token=os.getenv('huggingface_token')
    )

    output = ability_classifier.classify_ability(text_to_classify)
    output = output[0]

    return output

def chat_with_character_chatbot(message, history):
    character_chatbot = CharacterChatBot(
        "YOUR_HF_USERNAME/OnePiece_Luffy_Llama-3.1-8B",
        huggingface_token=os.getenv('huggingface_token')
    )

    output = character_chatbot.chat(message, history)
    # output is a dict like {"role": "assistant", "content": "..."} when using
    # transformers chat-style pipelines; fall back gracefully if it's a string.
    if isinstance(output, dict):
        output = output.get('content', '').strip()
    else:
        output = str(output).strip()
    return output


def main():
    with gr.Blocks() as iface:

        # Theme Classification Section
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>One Piece — Theme Classification (Zero-Shot Classifier)</h1>")
                with gr.Row():
                    with gr.Column():
                        plot = gr.BarPlot()
                    with gr.Column():
                        theme_list = gr.Textbox(
                            label="Themes",
                            placeholder="freedom, adventure, friendship, dreams, justice, sacrifice, loyalty, betrayal, family, courage"
                        )
                        theme_subtitles_path = gr.Textbox(label="Subtitles or Script Path")
                        save_path = gr.Textbox(label="Save Path")
                        get_themes_button = gr.Button("Get Themes")
                        get_themes_button.click(
                            get_themes,
                            inputs=[theme_list, theme_subtitles_path, save_path],
                            outputs=[plot]
                        )

        # Character Network Section
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>One Piece — Character Network (NER + Graph)</h1>")
                with gr.Row():
                    with gr.Column():
                        network_html = gr.HTML()
                    with gr.Column():
                        network_subtitles_path = gr.Textbox(label="Subtitles or Script Path")
                        ner_path = gr.Textbox(label="NER Save Path")
                        get_network_graph_button = gr.Button("Get Character Network")
                        get_network_graph_button.click(
                            get_character_network,
                            inputs=[network_subtitles_path, ner_path],
                            outputs=[network_html]
                        )

        # Ability Classification Section
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>One Piece — Ability Classification (Devil Fruit / Haki / Physical Technique)</h1>")
                with gr.Row():
                    with gr.Column():
                        text_classification_output = gr.Textbox(label="Ability Classification Output")
                    with gr.Column():
                        text_classification_model = gr.Textbox(label="Model Path")
                        text_classification_data_path = gr.Textbox(label="Data Path")
                        text_to_classify = gr.Textbox(label="Ability Description")
                        classify_text_button = gr.Button("Classify Ability")
                        classify_text_button.click(
                            classify_text,
                            inputs=[text_classification_model, text_classification_data_path, text_to_classify],
                            outputs=[text_classification_output]
                        )

        # Luffy Chatbot Section
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>One Piece — Luffy Chatbot</h1>")
                gr.ChatInterface(
                    chat_with_character_chatbot,
                    type="messages"
                )

    iface.launch(share=True)


if __name__ == '__main__':
    main()
