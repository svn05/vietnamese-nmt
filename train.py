"""Fine-tune NLLB-200 for Vietnamese-English-Japanese translation.

Fine-tunes Meta's NLLB-200 model on OPUS parallel corpora for
trilingual translation with beam search decoding.

Usage:
    python train.py
    python train.py --epochs 3 --batch-size 8 --lr 5e-5
"""

import argparse
import json
import os
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm


class TranslationDataset(Dataset):
    """Dataset for parallel sentence pairs."""

    def __init__(self, pairs, tokenizer, src_lang, tgt_lang, max_length=128):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        self.tokenizer.src_lang = self.src_lang

        source = self.tokenizer(
            pair["src"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        with self.tokenizer.as_target_tokenizer():
            target = self.tokenizer(
                pair["tgt"],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

        labels = target["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": source["input_ids"].squeeze(0),
            "attention_mask": source["attention_mask"].squeeze(0),
            "labels": labels,
        }


def load_pairs(filepath):
    """Load parallel pairs from JSONL file."""
    pairs = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            pairs.append(json.loads(line.strip()))
    return pairs


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    progress = tqdm(dataloader, desc="Training")
    for batch in progress:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * input_ids.size(0)
        progress.set_postfix(loss=loss.item())

    return total_loss / len(dataloader.dataset)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune NLLB for trilingual NMT")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model and tokenizer
    model_name = config["model"]["name"]
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    # Load training data
    data_dir = "data"
    vi_en_path = os.path.join(data_dir, "vi_en_train.jsonl")
    vi_ja_path = os.path.join(data_dir, "vi_ja_train.jsonl")

    if not os.path.exists(vi_en_path):
        print("Training data not found. Generating synthetic data...")
        from data.prepare_opus import generate_synthetic_data, save_data
        vi_en, vi_ja = generate_synthetic_data()
        save_data([{"src": p["vi"], "tgt": p["en"]} for p in vi_en], "vi_en_train.jsonl")
        save_data([{"src": p["vi"], "tgt": p["ja"]} for p in vi_ja], "vi_ja_train.jsonl")

    vi_en_pairs = load_pairs(vi_en_path)
    print(f"Loaded {len(vi_en_pairs)} vi-en pairs")

    # Language codes for NLLB
    lang_codes = config["languages"]
    src_lang = lang_codes["vietnamese"]["code"]
    tgt_lang = lang_codes["english"]["code"]

    # Create dataset
    train_dataset = TranslationDataset(
        vi_en_pairs, tokenizer, src_lang, tgt_lang,
        max_length=config["model"]["max_length"],
    )

    batch_size = args.batch_size or config["training"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Optimizer and scheduler
    epochs = args.epochs or config["training"]["epochs"]
    lr = args.lr or config["training"]["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config["training"]["weight_decay"])
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * config["training"]["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Training loop
    output_dir = config["output"]["model_dir"]
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nTraining for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Epoch {epoch}/{epochs} — Loss: {train_loss:.4f}")

        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    print(f"\nModel saved to {output_dir}")


if __name__ == "__main__":
    main()
