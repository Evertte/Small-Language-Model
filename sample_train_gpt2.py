import argparse
import torch

from train_gpt2 import GPT, GPTConfig


class CharTokenizer:
    def __init__(self, stoi):
        self.stoi = {str(k): int(v) for k, v in stoi.items()}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.unk_id = self.stoi.get(" ", 0)

    def encode(self, text):
        return [self.stoi.get(ch, self.unk_id) for ch in text]

    def decode(self, token_ids):
        return "".join(self.itos[int(i)] for i in token_ids)


def generate(model, idx, max_new_tokens, temperature=0.8, top_k=40, stop_ids=None):
    block_size = model.config.block_size
    if stop_ids:
        stop_ids = torch.tensor(stop_ids, dtype=torch.long, device=idx.device)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-5)

        if top_k is not None and top_k > 0:
            k = min(top_k, logits.size(-1))
            topk_vals, _ = torch.topk(logits, k=k, dim=-1)
            logits[logits < topk_vals[:, [-1]]] = -float("inf")

        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
        if stop_ids is not None and idx.size(1) >= stop_ids.numel():
            tail = idx[:, -stop_ids.numel():]
            if (tail == stop_ids.unsqueeze(0)).all(dim=1).all():
                break
    return idx


def main():
    parser = argparse.ArgumentParser(description="Sample from a train_gpt2.py checkpoint.")
    parser.add_argument("--ckpt", type=str, default="checkpoints/gpt2_chat.best.pt")
    parser.add_argument("--start", type=str, default="User: I feel overwhelmed\nAssistant:")
    parser.add_argument("--max_new_tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint = torch.load(args.ckpt, map_location="cpu")
    tokenizer_meta = checkpoint.get("tokenizer", {})
    tokenizer_type = tokenizer_meta.get("type", "gpt2")

    if tokenizer_type == "char":
        tokenizer = CharTokenizer(tokenizer_meta["stoi"])
        encode = tokenizer.encode
        decode = tokenizer.decode
    elif tokenizer_type == "gpt2":
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
        encode = enc.encode
        decode = enc.decode
    else:
        raise ValueError(f"Unsupported tokenizer type in checkpoint: {tokenizer_type}")

    model = GPT(GPTConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval().to(args.device)

    ids = encode(args.start)
    if len(ids) == 0:
        ids = [0]
    ids = ids[-model.config.block_size :]
    x = torch.tensor(ids, dtype=torch.long, device=args.device)[None, :]

    user_prefix = tokenizer_meta.get("user_prefix", "User:")
    stop_ids = encode(f"\n{user_prefix}")
    with torch.no_grad():
        y = generate(
            model=model,
            idx=x,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            stop_ids=stop_ids,
        )

    print(decode(y[0].tolist()))


if __name__ == "__main__":
    main()
