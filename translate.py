"""Translation with beam search decoding.

Translates text between Vietnamese, English, and Japanese using
the fine-tuned NLLB model with configurable beam search.

Usage:
    python translate.py --text "Xin chào" --src vi --tgt en
    python translate.py --text "Hello world" --src en --tgt vi
"""

import argparse
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# NLLB language codes
LANG_CODES = {
    "vi": "vie_Latn",
    "en": "eng_Latn",
    "ja": "jpn_Jpan",
    "vietnamese": "vie_Latn",
    "english": "eng_Latn",
    "japanese": "jpn_Jpan",
}

LANG_NAMES = {
    "vie_Latn": "Vietnamese",
    "eng_Latn": "English",
    "jpn_Jpan": "Japanese",
}


def load_model(model_dir="outputs/model"):
    """Load fine-tuned NLLB model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)
    except Exception:
        print("Fine-tuned model not found. Using base NLLB-200...")
        model_name = "facebook/nllb-200-distilled-600M"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    model.eval()
    return model, tokenizer, device


def translate(
    text, model, tokenizer, device,
    src_lang="vie_Latn", tgt_lang="eng_Latn",
    beam_size=5, max_length=128,
    length_penalty=1.0, no_repeat_ngram_size=3,
):
    """Translate text with beam search decoding.

    Args:
        text: Input text to translate.
        model: NLLB model.
        tokenizer: NLLB tokenizer.
        device: torch device.
        src_lang: Source language NLLB code.
        tgt_lang: Target language NLLB code.
        beam_size: Number of beams for beam search.
        max_length: Maximum output length.
        length_penalty: Beam search length penalty.
        no_repeat_ngram_size: Prevent n-gram repetition.

    Returns:
        Translated text string.
    """
    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    tgt_lang_id = tokenizer.convert_tokens_to_ids(tgt_lang)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tgt_lang_id,
            num_beams=beam_size,
            max_length=max_length,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=True,
        )

    translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated


def translate_batch(texts, model, tokenizer, device, src_lang, tgt_lang, beam_size=5):
    """Translate a batch of texts."""
    results = []
    for text in texts:
        result = translate(text, model, tokenizer, device, src_lang, tgt_lang, beam_size)
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description="Translate between Vietnamese, English, Japanese")
    parser.add_argument("--text", type=str, required=True, help="Text to translate")
    parser.add_argument("--src", type=str, default="vi", help="Source language (vi/en/ja)")
    parser.add_argument("--tgt", type=str, default="en", help="Target language (vi/en/ja)")
    parser.add_argument("--model-dir", type=str, default="outputs/model")
    parser.add_argument("--beam-size", type=int, default=5)
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.model_dir)

    src_code = LANG_CODES.get(args.src, args.src)
    tgt_code = LANG_CODES.get(args.tgt, args.tgt)

    result = translate(
        args.text, model, tokenizer, device,
        src_lang=src_code, tgt_lang=tgt_code,
        beam_size=args.beam_size,
    )

    src_name = LANG_NAMES.get(src_code, args.src)
    tgt_name = LANG_NAMES.get(tgt_code, args.tgt)

    print(f"\n{src_name}: {args.text}")
    print(f"{tgt_name}: {result}")


if __name__ == "__main__":
    main()
