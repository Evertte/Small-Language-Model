import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description="Remove training-only state from a train.py checkpoint.")
    parser.add_argument("--src", default="models/mental_health_chat/ckpt.pt")
    parser.add_argument("--dst", default="mental_health_infer.pt")
    parser.add_argument("--fp16", action="store_true", help="Store model tensors as float16 to reduce file size.")
    args = parser.parse_args()

    ckpt = torch.load(args.src, map_location="cpu")
    model_state = ckpt["model"]
    if args.fp16:
        model_state = {
            key: value.half() if torch.is_floating_point(value) else value
            for key, value in model_state.items()
        }

    small = {
        "model": model_state,
        "model_args": ckpt["model_args"],
        "iter_num": ckpt.get("iter_num"),
        "best_val_loss": ckpt.get("best_val_loss"),
        "config": ckpt.get("config", {}),
    }

    dst = Path(args.dst)
    torch.save(small, dst)
    print(f"saved {dst} ({dst.stat().st_size / (1024 ** 2):.1f} MB)")


if __name__ == "__main__":
    main()
