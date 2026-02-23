"""Language auto-detection for the trilingual translator.

Detects whether input text is Vietnamese, English, or Japanese
to automatically select the source language for translation.

Usage:
    python detect_language.py --text "Xin chào các bạn"
"""

import argparse


def detect_language(text):
    """Detect language of input text.

    Uses character-based heuristics for fast detection:
    - Japanese: contains Hiragana, Katakana, or CJK characters
    - Vietnamese: contains Vietnamese diacritical marks
    - English: default fallback

    Args:
        text: Input text string.

    Returns:
        Language code ('vi', 'en', or 'ja') and confidence score.
    """
    if not text.strip():
        return "en", 0.0

    text = text.strip()

    # Count character types
    n_chars = len(text.replace(" ", ""))
    if n_chars == 0:
        return "en", 0.0

    n_cjk = 0
    n_hiragana = 0
    n_katakana = 0
    n_vietnamese_diacritics = 0
    n_latin = 0

    # Vietnamese-specific characters
    vn_chars = set("ăắằẳẵặâấầẩẫậêếềểễệôốồổỗộơớờởỡợưứừửữựđĂẮẰẲẴẶÂẤẦẨẪẬÊẾỀỂỄỆÔỐỒỔỖỘƠỚỜỞỠỢƯỨỪỬỮỰĐ")
    vn_tones = set("àáảãạèéẻẽẹìíỉĩịòóỏõọùúủũụỳýỷỹỵÀÁẢÃẠÈÉẺẼẸÌÍỈĨỊÒÓỎÕỌÙÚỦŨỤỲÝỶỸỴ")

    for char in text:
        if char == " ":
            continue

        cp = ord(char)

        # Japanese detection
        if 0x3040 <= cp <= 0x309F:
            n_hiragana += 1
        elif 0x30A0 <= cp <= 0x30FF:
            n_katakana += 1
        elif 0x4E00 <= cp <= 0x9FFF:
            n_cjk += 1
        elif char in vn_chars or char in vn_tones:
            n_vietnamese_diacritics += 1
        elif char.isascii() and char.isalpha():
            n_latin += 1

    # Decision logic
    n_japanese = n_hiragana + n_katakana + n_cjk

    if n_japanese > n_chars * 0.1:
        confidence = min(n_japanese / n_chars, 1.0)
        return "ja", confidence

    if n_vietnamese_diacritics > 0:
        confidence = min((n_vietnamese_diacritics / n_chars) * 5, 1.0)
        return "vi", max(confidence, 0.7)

    # Check for Vietnamese words without diacritics
    vn_words = {"xin", "chao", "cam", "ban", "toi", "cac", "nhung", "nhu", "va", "la",
                "cua", "duoc", "khong", "nay", "mot", "co", "trong", "cho", "voi"}
    words = set(text.lower().split())
    vn_word_count = len(words & vn_words)
    if vn_word_count >= 2:
        return "vi", 0.6

    # Default to English
    if n_latin > 0:
        return "en", min(n_latin / n_chars, 1.0)

    return "en", 0.5


def main():
    parser = argparse.ArgumentParser(description="Detect language of text")
    parser.add_argument("--text", type=str, required=True)
    args = parser.parse_args()

    lang, confidence = detect_language(args.text)
    lang_names = {"vi": "Vietnamese", "en": "English", "ja": "Japanese"}
    print(f"Text: {args.text}")
    print(f"Detected: {lang_names[lang]} ({confidence:.2%})")


if __name__ == "__main__":
    main()
