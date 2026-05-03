import argparse
import pickle

import torch
import tiktoken

from model import GPT, GPTConfig


def generate(model, idx, max_new_tokens, temperature=0.8, top_k=40):
    block_size = model.config.block_size
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
    return idx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="out/ckpt.pt")
    p.add_argument("--start", type=str, default="User: Hi\nAssistant:")
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_k", type=int, default=30)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_samples", type=int, default=1)
    args = p.parse_args()

    checkpoint = torch.load(args.ckpt, map_location=args.device)
    model_args = checkpoint["model_args"]

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval().to(args.device)

    enc = tiktoken.get_encoding("gpt2")
    start_ids = enc.encode(args.start)
    if len(start_ids) == 0:
        start_ids = [enc.eot_token]

    x = torch.tensor(start_ids, dtype=torch.long, device=args.device).unsqueeze(0)
    x = x.repeat(args.num_samples, 1)

    with torch.no_grad():
        y = generate(
            model=model,
            idx=x,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )

    for i in range(args.num_samples):
        print("\n--- sample", i + 1, "---")
        print(enc.decode(y[i].tolist()))


if __name__ == "__main__":
    main()
