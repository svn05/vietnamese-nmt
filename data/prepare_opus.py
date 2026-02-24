"""Download and prepare OPUS parallel corpora for NMT training.

Loads Vietnamese-English and Vietnamese-Japanese parallel data
from OPUS datasets via HuggingFace.

Usage:
    python data/prepare_opus.py
    python data/prepare_opus.py --max-samples 10000
"""

import argparse
import os
import json
import random
from datasets import load_dataset


DATA_DIR = os.path.dirname(__file__)

# Sample parallel data for quick testing
SAMPLE_VI_EN = [
    {"vi": "Xin chào, tôi tên là San.", "en": "Hello, my name is San."},
    {"vi": "Hôm nay thời tiết rất đẹp.", "en": "The weather is very nice today."},
    {"vi": "Tôi đang học ngành kỹ thuật cơ khí.", "en": "I am studying mechanical engineering."},
    {"vi": "Việt Nam là một đất nước xinh đẹp.", "en": "Vietnam is a beautiful country."},
    {"vi": "Tôi thích ăn phở và bánh mì.", "en": "I like eating pho and banh mi."},
    {"vi": "Trí tuệ nhân tạo đang thay đổi thế giới.", "en": "Artificial intelligence is changing the world."},
    {"vi": "Đại học Toronto là một trong những trường tốt nhất.", "en": "The University of Toronto is one of the best schools."},
    {"vi": "Tôi muốn trở thành kỹ sư phần mềm.", "en": "I want to become a software engineer."},
    {"vi": "Cảm ơn bạn rất nhiều.", "en": "Thank you very much."},
    {"vi": "Chúng tôi đang phát triển một ứng dụng mới.", "en": "We are developing a new application."},
]

SAMPLE_VI_JA = [
    {"vi": "Xin chào.", "ja": "こんにちは。"},
    {"vi": "Cảm ơn bạn.", "ja": "ありがとうございます。"},
    {"vi": "Tôi là sinh viên.", "ja": "私は学生です。"},
    {"vi": "Tôi đang học tiếng Nhật.", "ja": "私は日本語を勉強しています。"},
    {"vi": "Hôm nay trời đẹp quá.", "ja": "今日はいい天気ですね。"},
    {"vi": "Tôi thích ăn sushi.", "ja": "私は寿司が好きです。"},
    {"vi": "Nhật Bản là một đất nước tuyệt vời.", "ja": "日本は素晴らしい国です。"},
    {"vi": "Tôi muốn đi du lịch Tokyo.", "ja": "東京に旅行したいです。"},
    {"vi": "Đây là quyển sách rất hay.", "ja": "これはとても良い本です。"},
    {"vi": "Chúc bạn một ngày tốt lành.", "ja": "良い一日を。"},
]


def load_opus_data(src_lang, tgt_lang, max_samples=50000):
    """Load OPUS parallel corpus from HuggingFace.

    Args:
        src_lang: Source language code (e.g., 'vi').
        tgt_lang: Target language code (e.g., 'en').
        max_samples: Maximum number of sentence pairs.

    Returns:
        List of dicts with 'src' and 'tgt' keys.
    """
    lang_pair = f"{src_lang}-{tgt_lang}"
    try:
        dataset = load_dataset("Helsinki-NLP/opus-100", lang_pair, split="train")
    except Exception:
        # Try reversed pair
        lang_pair = f"{tgt_lang}-{src_lang}"
        try:
            dataset = load_dataset("Helsinki-NLP/opus-100", lang_pair, split="train")
        except Exception as e:
            print(f"Could not load OPUS data for {src_lang}-{tgt_lang}: {e}")
            return []

    pairs = []
    for item in dataset:
        translation = item.get("translation", item)
        if src_lang in translation and tgt_lang in translation:
            pairs.append({"src": translation[src_lang], "tgt": translation[tgt_lang]})
        if len(pairs) >= max_samples:
            break

    print(f"Loaded {len(pairs)} {src_lang}-{tgt_lang} pairs")
    return pairs


def generate_synthetic_data(n_samples=5000):
    """Generate synthetic parallel data for quick testing.

    Expands sample sentences with simple variations.

    Returns:
        vi_en_pairs, vi_ja_pairs
    """
    random.seed(42)

    vi_en = []
    vi_ja = []

    # Expand samples with minor variations
    for _ in range(n_samples // len(SAMPLE_VI_EN)):
        for pair in SAMPLE_VI_EN:
            vi_en.append(pair.copy())
    for _ in range(n_samples // len(SAMPLE_VI_JA)):
        for pair in SAMPLE_VI_JA:
            vi_ja.append(pair.copy())

    random.shuffle(vi_en)
    random.shuffle(vi_ja)

    return vi_en[:n_samples], vi_ja[:n_samples]


def save_data(pairs, filename, src_key="src", tgt_key="tgt"):
    """Save parallel data as JSONL."""
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        for pair in pairs:
            json.dump(pair, f, ensure_ascii=False)
            f.write("\n")
    print(f"Saved {len(pairs)} pairs to {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Prepare OPUS parallel data")
    parser.add_argument("--max-samples", type=int, default=50000)
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    args = parser.parse_args()

    if args.synthetic:
        print("Generating synthetic parallel data...")
        vi_en, vi_ja = generate_synthetic_data()
        save_data([{"src": p["vi"], "tgt": p["en"]} for p in vi_en], "vi_en_train.jsonl")
        save_data([{"src": p["vi"], "tgt": p["ja"]} for p in vi_ja], "vi_ja_train.jsonl")
    else:
        print("Loading OPUS parallel corpora...")
        vi_en = load_opus_data("vi", "en", args.max_samples)
        if vi_en:
            save_data(vi_en, "vi_en_train.jsonl")

        vi_ja = load_opus_data("vi", "ja", args.max_samples)
        if vi_ja:
            save_data(vi_ja, "vi_ja_train.jsonl")


if __name__ == "__main__":
    main()
