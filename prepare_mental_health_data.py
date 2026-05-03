import argparse
import os
import pickle
import random

import numpy as np
import tiktoken
from datasets import load_dataset


def clean(s: str) -> str:
    return " ".join((s or "").replace("\r", " ").replace("\n", " ").split())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", type=str, default="Amod/mental_health_counseling_conversations")
    parser.add_argument("--out_dir", type=str, default=os.path.join("data", "mental_health_chat"))
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--repeat", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    ds = load_dataset(args.dataset_id, split="train")

    # Build conversation-formatted text so the model learns turn structure.
    samples = []
    for ex in ds:
        user = clean(ex.get("Context", ""))
        assistant = clean(ex.get("Response", ""))
        if not user or not assistant:
            continue
        samples.append(f"User: {user}\nAssistant: {assistant}\n\n")

    if len(samples) < 100:
        raise RuntimeError(f"too few usable samples: {len(samples)}")

    random.shuffle(samples)
    # Expand epochs-in-a-file for scratch training stability on small corpora.
    if args.repeat > 1:
        samples = samples * args.repeat
        random.shuffle(samples)
    split_idx = int(args.train_ratio * len(samples))
    train_text = "".join(samples[:split_idx])
    val_text = "".join(samples[split_idx:])

    enc = tiktoken.get_encoding("gpt2")
    train_ids = enc.encode_ordinary(train_text)
    val_ids = enc.encode_ordinary(val_text)

    # Karpathy train.py expects uint16 memmaps.
    train_arr = np.array(train_ids, dtype=np.uint16)
    val_arr = np.array(val_ids, dtype=np.uint16)

    train_path = os.path.join(args.out_dir, "train.bin")
    val_path = os.path.join(args.out_dir, "val.bin")
    meta_path = os.path.join(args.out_dir, "meta.pkl")

    train_arr.tofile(train_path)
    val_arr.tofile(val_path)

    meta = {
        "vocab_size": enc.n_vocab,
        "tokenizer": "gpt2",
        "dataset": args.dataset_id,
        "num_samples": len(samples),
        "train_ratio": args.train_ratio,
        "repeat": args.repeat,
    }
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)

    print(f"saved: {train_path} ({train_arr.size:,} tokens)")
    print(f"saved: {val_path} ({val_arr.size:,} tokens)")
    print(f"saved: {meta_path}")


if __name__ == "__main__":
    main()
