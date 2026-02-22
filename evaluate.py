"""Evaluate translation quality with BLEU scores.

Usage:
    python evaluate.py
"""

import argparse
import json
import os
import sacrebleu
import yaml

from translate import load_model, translate, LANG_CODES


def evaluate_bleu(model, tokenizer, device, test_pairs, src_lang, tgt_lang):
    """Compute BLEU score on test pairs.

    Args:
        model: Translation model.
        tokenizer: Tokenizer.
        device: torch device.
        test_pairs: List of {"src": ..., "tgt": ...} dicts.
        src_lang: Source NLLB code.
        tgt_lang: Target NLLB code.

    Returns:
        BLEU score object and list of translations.
    """
    references = []
    hypotheses = []

    for pair in test_pairs:
        translation = translate(pair["src"], model, tokenizer, device, src_lang, tgt_lang)
        hypotheses.append(translation)
        references.append(pair["tgt"])

    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return bleu, hypotheses


def main():
    parser = argparse.ArgumentParser(description="Evaluate NMT with BLEU")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--model-dir", type=str, default="outputs/model")
    parser.add_argument("--max-samples", type=int, default=100)
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.model_dir)

    # Load test data
    data_dir = "data"
    results = {}

    for pair_name, src, tgt in [("vi→en", "vi", "en"), ("vi→ja", "vi", "ja")]:
        filepath = os.path.join(data_dir, f"vi_{tgt}_train.jsonl")
        if not os.path.exists(filepath):
            continue

        pairs = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                pairs.append(json.loads(line.strip()))
                if len(pairs) >= args.max_samples:
                    break

        src_code = LANG_CODES[src]
        tgt_code = LANG_CODES[tgt]

        print(f"\nEvaluating {pair_name} ({len(pairs)} samples)...")
        bleu, translations = evaluate_bleu(model, tokenizer, device, pairs, src_code, tgt_code)

        print(f"  BLEU: {bleu.score:.2f}")
        print(f"  Details: {bleu}")
        results[pair_name] = bleu.score

        # Show examples
        print(f"\n  Sample translations:")
        for i in range(min(3, len(pairs))):
            print(f"    Source: {pairs[i]['src']}")
            print(f"    Target: {pairs[i]['tgt']}")
            print(f"    Predicted: {translations[i]}")
            print()

    print("\n" + "=" * 40)
    print("BLEU Score Summary")
    print("=" * 40)
    for pair, score in results.items():
        print(f"  {pair}: {score:.2f}")


if __name__ == "__main__":
    main()
