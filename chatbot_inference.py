import re
from pathlib import Path

import torch
import tiktoken

from model import GPT, GPTConfig


class CharTokenizer:
    def __init__(self, stoi: dict):
        self.stoi = {str(k): int(v) for k, v in stoi.items()}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.unk_id = self.stoi.get(" ", 0)

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(ch, self.unk_id) for ch in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.itos.get(int(i), "") for i in token_ids)


CRISIS_RESPONSE = (
    "I am not able to provide emergency or crisis support. If you might hurt "
    "yourself or someone else, call or text 988 in the U.S. and Canada, call "
    "911 or your local emergency number, or go to the nearest emergency room. "
    "If possible, contact someone you trust and stay with them while you get help."
)

CRISIS_PATTERNS = [
    r"\bkill myself\b",
    r"\bsuicide\b",
    r"\bsuicidal\b",
    r"\bend my life\b",
    r"\bhurt myself\b",
    r"\bself[- ]?harm\b",
    r"\boverdose\b",
    r"\bwant to die\b",
    r"\bcan't go on\b",
    r"\bhurt someone\b",
    r"\bkill someone\b",
]


def pick_device(preferred: str = "auto") -> str:
    if preferred != "auto":
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def is_crisis_message(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in CRISIS_PATTERNS)


def load_chat_model(ckpt_path: str | Path, device: str = "auto"):
    device = pick_device(device)
    checkpoint = torch.load(str(ckpt_path), map_location=device)
    model_args = checkpoint.get("model_args", checkpoint.get("config"))
    if model_args is None:
        raise ValueError("Checkpoint must contain either 'model_args' or 'config'.")

    model = GPT(GPTConfig(**model_args))
    state_dict = checkpoint.get("model", checkpoint.get("model_state_dict"))
    if state_dict is None:
        raise ValueError("Checkpoint must contain either 'model' or 'model_state_dict'.")
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix) :]] = state_dict.pop(key)

    model.load_state_dict(state_dict)
    model.eval().to(device)

    tokenizer_meta = checkpoint.get("tokenizer", {})
    if tokenizer_meta.get("type") == "char":
        enc = CharTokenizer(tokenizer_meta["stoi"])
    else:
        enc = tiktoken.get_encoding("gpt2")
    return model, enc, device


def build_prompt(history: list[dict], user_message: str, max_turns: int = 4) -> str:
    prompt_parts = []
    for message in history[-max_turns * 2 :]:
        role = message.get("role")
        content = clean_text(message.get("content", ""))
        if not content:
            continue
        if role == "user":
            prompt_parts.append(f"User: {content}")
        elif role == "assistant":
            prompt_parts.append(f"Assistant: {content}")
    prompt_parts.append(f"User: {clean_text(user_message)}")
    prompt_parts.append("Assistant:")
    return "\n".join(prompt_parts)


def clean_text(text: str) -> str:
    return " ".join((text or "").replace("\r", " ").split())


@torch.no_grad()
def generate_tokens(
    model,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 0.7,
    top_k: int = 40,
    repetition_penalty: float = 1.15,
):
    block_size = model.config.block_size
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-5)

        if repetition_penalty > 1.0:
            recent_tokens = set(idx[0, -128:].tolist())
            for token_id in recent_tokens:
                if logits[0, token_id] < 0:
                    logits[0, token_id] *= repetition_penalty
                else:
                    logits[0, token_id] /= repetition_penalty

        if top_k is not None and top_k > 0:
            k = min(top_k, logits.size(-1))
            topk_vals, _ = torch.topk(logits, k=k, dim=-1)
            logits[logits < topk_vals[:, [-1]]] = -float("inf")

        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def extract_reply(prompt: str, decoded: str) -> str:
    reply = decoded[len(prompt) :] if decoded.startswith(prompt) else decoded
    for stop_marker in ("\nUser:", "\n\nUser:", "\nAssistant:"):
        if stop_marker in reply:
            reply = reply.split(stop_marker, 1)[0]
    reply = reply.strip()
    return reply or "I am not sure how to answer that. Please rephrase your question."


@torch.no_grad()
def stream_reply(
    model,
    enc,
    device: str,
    history: list[dict],
    user_message: str,
    max_new_tokens: int = 180,
    temperature: float = 0.45,
    top_k: int = 25,
    repetition_penalty: float = 1.15,
):
    if is_crisis_message(user_message):
        yield CRISIS_RESPONSE
        return

    prompt = build_prompt(history, user_message)
    start_ids = enc.encode(prompt)
    start_ids = start_ids[-model.config.block_size :]
    idx = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)
    emitted = ""
    sent_len = 0

    for _ in range(max_new_tokens):
        before = idx.size(1)
        idx = generate_tokens(
            model=model,
            idx=idx,
            max_new_tokens=1,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
        token_text = enc.decode(idx[0, before:].tolist())
        emitted += token_text

        for stop_marker in ("\nUser:", "\n\nUser:", "\nAssistant:"):
            if stop_marker in emitted:
                visible = emitted.split(stop_marker, 1)[0]
                chunk = visible[sent_len:]
                if chunk:
                    yield chunk
                return

        yield token_text
        sent_len = len(emitted)


def generate_reply(
    model,
    enc,
    device: str,
    history: list[dict],
    user_message: str,
    max_new_tokens: int = 180,
    temperature: float = 0.45,
    top_k: int = 25,
    repetition_penalty: float = 1.15,
) -> str:
    if is_crisis_message(user_message):
        return CRISIS_RESPONSE

    prompt = build_prompt(history, user_message)
    start_ids = enc.encode(prompt)
    start_ids = start_ids[-model.config.block_size :]
    x = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)

    y = generate_tokens(
        model=model,
        idx=x,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )
    decoded = enc.decode(y[0].tolist())
    return extract_reply(prompt, decoded)
