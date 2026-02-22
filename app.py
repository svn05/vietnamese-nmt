"""Trilingual Gradio interface for Vietnamese-English-Japanese translation.

Provides a web UI with language auto-detection and beam search translation.

Usage:
    python app.py
"""

import gradio as gr
from translate import load_model, translate, LANG_CODES, LANG_NAMES
from detect_language import detect_language


# Load model at startup
model, tokenizer, device = load_model()


def translate_text(text, src_lang, tgt_lang, beam_size=5, auto_detect=True):
    """Translate text with optional language auto-detection.

    Args:
        text: Input text.
        src_lang: Source language name.
        tgt_lang: Target language name.
        beam_size: Beam search width.
        auto_detect: Whether to auto-detect source language.

    Returns:
        Translated text and detected language info.
    """
    if not text.strip():
        return "", "No input"

    # Auto-detect source language
    detected_info = ""
    if auto_detect and src_lang == "Auto-detect":
        lang_code, confidence = detect_language(text)
        lang_map = {"vi": "Vietnamese", "en": "English", "ja": "Japanese"}
        src_lang = lang_map[lang_code]
        detected_info = f"Detected: {src_lang} ({confidence:.0%})"

    # Get NLLB codes
    src_code = LANG_CODES.get(src_lang.lower(), "vie_Latn")
    tgt_code = LANG_CODES.get(tgt_lang.lower(), "eng_Latn")

    if src_code == tgt_code:
        return text, "Source and target languages are the same"

    result = translate(
        text, model, tokenizer, device,
        src_lang=src_code, tgt_lang=tgt_code,
        beam_size=beam_size,
    )

    return result, detected_info


# Gradio interface
languages = ["Auto-detect", "Vietnamese", "English", "Japanese"]
target_languages = ["Vietnamese", "English", "Japanese"]

examples = [
    ["Xin chào, tôi tên là San. Tôi đang học tại Đại học Toronto.", "Auto-detect", "English", 5],
    ["Artificial intelligence is changing the world rapidly.", "Auto-detect", "Vietnamese", 5],
    ["こんにちは、お元気ですか。", "Auto-detect", "Vietnamese", 5],
    ["Việt Nam là một đất nước xinh đẹp với nhiều danh lam thắng cảnh.", "Vietnamese", "Japanese", 5],
    ["I want to learn Vietnamese and Japanese.", "English", "Vietnamese", 5],
]

demo = gr.Interface(
    fn=translate_text,
    inputs=[
        gr.Textbox(label="Input Text", placeholder="Enter text to translate...", lines=4),
        gr.Dropdown(choices=languages, value="Auto-detect", label="Source Language"),
        gr.Dropdown(choices=target_languages, value="English", label="Target Language"),
        gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Beam Size"),
    ],
    outputs=[
        gr.Textbox(label="Translation", lines=4),
        gr.Textbox(label="Info"),
    ],
    title="Vietnamese ↔ English ↔ Japanese Neural Machine Translation",
    description=(
        "Trilingual translation powered by **Meta's NLLB-200**, fine-tuned on OPUS parallel corpora. "
        "Supports Vietnamese ↔ English ↔ Japanese with beam search decoding and language auto-detection."
    ),
    examples=examples,
    theme=gr.themes.Soft(),
)


if __name__ == "__main__":
    demo.launch(share=False)
