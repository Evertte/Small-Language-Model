import argparse
from dataclasses import dataclass
import json
import math
import os
import re
import time
import torch
import torch.nn as nn
from torch.nn import functional as F

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    """The attention mechanism is like two vectors side by side that work indpendently and their output is concatenated together. The query vector is multiplied to the key vector and value vector to get the attention weights, which are then multiplied to the value vector to get the output.
    the query, key and value are then split into multiple heads, where each head is a separate attention mechanism that operates on a subset of the embedding dimensions. The output of each head is then concatenated together and passed through a final linear layer to produce the final output of the attention mechanism.
    the transpose makes sure the model focus on pre-tokens and not post-tokens. The scaled dot product attention is a more efficient way to compute the attention weights, which is used in the original Transformer paper. The is_causal=True argument ensures that the model only attends to previous tokens and not future tokens, which is important for autoregressive language modeling."""
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x) # there are 124M tokens and each token at this attention has 3 layers, query, key and value. The query is multiplied to the key here
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y
    
class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh') #non-normal activation
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
    
class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x)) # self-attention with residual connection where the tokens talk to each other and the output is added back to the input 
        x = x + self.mlp(self.ln_2(x)) # multi-layer perceptron where each token is processed independently and the output is added back to the input
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        """match the architecture of the original GPT paper, 
        which consists of a stack of transformer blocks, 
        followed by a final layer norm and a linear layer for the output logits. The transformer block is a stack of transformer layers, 
        where each transformer layer is a stack of transformer blocks, each containing a multi-head self-attention mechanism and a feed-forward neural network.
          The input to the model is first passed through an embedding layer that converts token indices into dense vectors, and then through a positional encoding 
          layer that adds information about the position of each token in the sequence. Finally, the output of the transformer blocks is passed through a linear layer
            to produce logits for each token in the vocabulary"""
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, loss_mask=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            per_token_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                reduction='none',
            ).view(B, T)
            if loss_mask is not None:
                mask = loss_mask.float()
                denom = mask.sum()
                if denom.item() > 0:
                    loss = (per_token_loss * mask).sum() / denom
                else:
                    loss = per_token_loss.mean() * 0.0
            else:
                loss = per_token_loss.mean()
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type (hyper-parameters)
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    # def configure_optimizers(self, weight_decay, learning_rate, device_type):
    #     # start with all of the candidate parameters (that require grad)
    #     param_dict = {pn: p for pn, p in self.named_parameters()}
    #     param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    #     # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    #     # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    #     decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    #     nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    #     optim_groups = [
    #         {'params': decay_params, 'weight_decay': weight_decay},
    #         {'params': nodecay_params, 'weight_decay': 0.0}
    #     ]
    #     num_decay_params = sum(p.numel() for p in decay_params)
    #     num_nodecay_params = sum(p.numel() for p in nodecay_params)
    #     if master_process:
    #         print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    #         print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    #     # Create AdamW optimizer and use the fused version if it is available
    #     fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    #     use_fused = fused_available and device_type == "cuda"
    #     if master_process:
    #         print(f"using fused AdamW: {use_fused}")
    #     optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
    #     return optimizer

#----------------------------------------------------------------------------------------------
if __name__ == '__main__':
    class CharTokenizer:
        def __init__(self, text=None, stoi=None):
            if stoi is not None:
                self.stoi = {str(k): int(v) for k, v in stoi.items()}
            else:
                if text is None:
                    raise ValueError("CharTokenizer requires either `text` or `stoi`.")
                chars = sorted(list(set(text)))
                self.stoi = {ch: i for i, ch in enumerate(chars)}
            self.itos = {i: ch for ch, i in self.stoi.items()}
            self.unk_id = self.stoi.get(" ", 0)

        @property
        def vocab_size(self):
            return len(self.stoi)

        def encode(self, text):
            return [self.stoi.get(ch, self.unk_id) for ch in text]

        def decode(self, token_ids):
            return "".join(self.itos[i] for i in token_ids)

    class TokenDatasetLite:
        def __init__(self, tokens, block_size, target_mask=None):
            self.tokens = tokens
            self.block_size = block_size
            self.target_mask = target_mask
            if len(self.tokens) <= block_size + 1:
                raise ValueError(
                    f"Not enough tokens ({len(self.tokens)}) for block_size={block_size}. "
                    "Use a smaller block_size or more text."
                )
            if self.target_mask is not None and len(self.target_mask) != len(self.tokens):
                raise ValueError("target_mask length must match tokens length")

        def next_batch(self, batch_size, device, require_target=False):
            max_start = len(self.tokens) - self.block_size - 1
            starts = torch.randint(0, max_start, (batch_size,))
            offsets = torch.arange(self.block_size)
            x = self.tokens[starts[:, None] + offsets[None, :]]
            y = self.tokens[starts[:, None] + offsets[None, :] + 1]
            if self.target_mask is None:
                y_mask = torch.ones_like(y)
            else:
                y_mask = self.target_mask[starts[:, None] + offsets[None, :] + 1]

            # For assistant-only tuning, avoid wasting batches that have no supervised tokens.
            if require_target and y_mask.sum().item() == 0:
                for _ in range(16):
                    starts = torch.randint(0, max_start, (batch_size,))
                    x = self.tokens[starts[:, None] + offsets[None, :]]
                    y = self.tokens[starts[:, None] + offsets[None, :] + 1]
                    y_mask = self.target_mask[starts[:, None] + offsets[None, :] + 1]
                    if y_mask.sum().item() > 0:
                        break
            return x.to(device), y.to(device), y_mask.to(device)

    def build_assistant_char_target_mask(text, assistant_prefix, user_prefix):
        mask = torch.zeros(len(text), dtype=torch.long)
        pattern = re.compile(
            rf"{re.escape(assistant_prefix)}(.*?)(?=(\n{re.escape(user_prefix)})|\Z)",
            re.DOTALL,
        )
        matches = list(pattern.finditer(text))
        for m in matches:
            start = m.start(1)
            end = m.end(1)
            if start < end:
                mask[start:end] = 1
        return mask, len(matches)

    def build_assistant_token_ids_and_mask(text, encode, assistant_prefix, user_prefix):
        token_ids = []
        mask_ids = []
        cursor = 0
        assistant_spans = 0
        pattern = re.compile(
            rf"{re.escape(assistant_prefix)}(.*?)(?=(\n{re.escape(user_prefix)})|\Z)",
            re.DOTALL,
        )
        for match in pattern.finditer(text):
            prefix_end = match.start(1)
            assistant_end = match.end(1)

            context_ids = encode(text[cursor:prefix_end])
            token_ids.extend(context_ids)
            mask_ids.extend([0] * len(context_ids))

            assistant_ids = encode(text[prefix_end:assistant_end])
            token_ids.extend(assistant_ids)
            mask_ids.extend([1] * len(assistant_ids))

            assistant_spans += 1
            cursor = assistant_end

        tail_ids = encode(text[cursor:])
        token_ids.extend(tail_ids)
        mask_ids.extend([0] * len(tail_ids))

        if not token_ids:
            token_ids = encode(text)
            mask_ids = [0] * len(token_ids)
        return token_ids, torch.tensor(mask_ids, dtype=torch.long), assistant_spans

    def estimate_loss(model, train_data, val_data, batch_size, eval_iters, device, require_target):
        model.eval()
        out = {}
        with torch.no_grad():
            for split, dataset in (("train", train_data), ("val", val_data)):
                losses = torch.zeros(eval_iters)
                for k in range(eval_iters):
                    x, y, y_mask = dataset.next_batch(
                        batch_size,
                        device,
                        require_target=require_target,
                    )
                    _, loss = model(x, y, loss_mask=y_mask)
                    losses[k] = loss.item()
                out[split] = losses.mean().item()
        model.train()
        return out

    def generate(model, idx, block_size, max_new_tokens, temperature, top_k, stop_ids=None):
        if stop_ids is not None and len(stop_ids) > 0:
            stop_ids = torch.tensor(stop_ids, dtype=torch.long, device=idx.device)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None and top_k > 0:
                k = min(top_k, logits.size(-1))
                topk_vals, _ = torch.topk(logits, k=k, dim=-1)
                logits[logits < topk_vals[:, [-1]]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            if stop_ids is not None and idx.size(1) >= stop_ids.numel():
                tail = idx[:, -stop_ids.numel():]
                if (tail == stop_ids.unsqueeze(0)).all(dim=1).all():
                    break
        return idx

    parser = argparse.ArgumentParser(description="Train a GPT-2-style causal language model on local text.")
    parser.add_argument("--data_path", type=str, default="input.txt")
    parser.add_argument("--tokenizer", type=str, default="gpt2", choices=["char", "gpt2"])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--eval_iters", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--min_lr", type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--lr_decay_steps", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--n_layer", type=int, default=12)
    parser.add_argument("--n_head", type=int, default=12)
    parser.add_argument("--n_embd", type=int, default=768)
    parser.add_argument("--resume_from", type=str, default="")
    parser.add_argument("--save_path", type=str, default="checkpoints/tiny_shakespeare.pt")
    parser.add_argument("--best_save_path", type=str, default="")
    parser.add_argument("--no_checkpoint", action="store_true")
    parser.add_argument("--assistant_only_loss", action="store_true")
    parser.add_argument("--assistant_prefix", type=str, default="Assistant:")
    parser.add_argument("--user_prefix", type=str, default="User:")
    parser.add_argument("--stop_at_user", action="store_true")
    parser.add_argument("--start_text", type=str, default="User: Hi\nAssistant:")
    parser.add_argument("--generate_tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    print(f"using device: {device}")

    resume_checkpoint = None
    if args.resume_from:
        resume_checkpoint = torch.load(args.resume_from, map_location="cpu")
        print(f"resuming from checkpoint: {args.resume_from}")

    if args.n_embd % args.n_head != 0 and not resume_checkpoint:
        raise ValueError("n_embd must be divisible by n_head")

    with open(args.data_path, "r", encoding="utf-8") as f:
        text = f.read()
    if len(text.strip()) == 0:
        raise ValueError(f"Dataset file is empty: {args.data_path}")

    tokenizer_meta = {
        "type": args.tokenizer,
        "assistant_prefix": args.assistant_prefix,
        "user_prefix": args.user_prefix,
    }
    if args.tokenizer == "char":
        if (
            resume_checkpoint is not None
            and "tokenizer" in resume_checkpoint
            and resume_checkpoint["tokenizer"].get("type") == "char"
            and "stoi" in resume_checkpoint["tokenizer"]
        ):
            tokenizer = CharTokenizer(stoi=resume_checkpoint["tokenizer"]["stoi"])
            print("using tokenizer mapping from checkpoint")
        else:
            tokenizer = CharTokenizer(text=text)
        encode = tokenizer.encode
        decode = tokenizer.decode
        vocab_size = tokenizer.vocab_size
        tokenizer_meta["stoi"] = tokenizer.stoi
    else:
        try:
            import tiktoken
            enc = tiktoken.get_encoding("gpt2")
        except Exception as e:
            raise RuntimeError(
                "Failed to initialize GPT-2 tokenizer (tiktoken). "
                "Use --tokenizer char or install tiktoken in an environment with internet."
            ) from e
        encode = enc.encode
        decode = enc.decode
        vocab_size = enc.n_vocab

    if resume_checkpoint is not None and "config" in resume_checkpoint:
        data_block_size = int(resume_checkpoint["config"]["block_size"])
    else:
        data_block_size = args.block_size

    conversation_format = (args.user_prefix in text) and (args.assistant_prefix in text)
    assistant_only_loss = args.assistant_only_loss or conversation_format

    token_target_mask = None
    assistant_spans = 0
    if assistant_only_loss and args.tokenizer == "gpt2":
        token_ids, token_target_mask, assistant_spans = build_assistant_token_ids_and_mask(
            text=text,
            encode=encode,
            assistant_prefix=args.assistant_prefix,
            user_prefix=args.user_prefix,
        )
    else:
        token_ids = encode(text)
    if len(token_ids) < data_block_size + 2:
        raise ValueError(
            f"Not enough tokens ({len(token_ids)}) for block_size={data_block_size}. "
            "Use a smaller block_size or a larger dataset."
        )

    if args.tokenizer == "char" and conversation_format:
        split = int(args.train_ratio * len(text))
        next_user = text.find(f"\n{args.user_prefix}", split)
        if next_user != -1:
            split = next_user + 1
    else:
        split = int(args.train_ratio * len(token_ids))

    split = max(1, min(split, len(token_ids) - 1))
    train_tokens = torch.tensor(token_ids[:split], dtype=torch.long)
    val_tokens = torch.tensor(token_ids[split:], dtype=torch.long)

    train_target_mask = None
    val_target_mask = None
    if assistant_only_loss:
        if args.tokenizer == "char":
            char_mask, assistant_spans = build_assistant_char_target_mask(
                text=text,
                assistant_prefix=args.assistant_prefix,
                user_prefix=args.user_prefix,
            )
            train_target_mask = char_mask[:split]
            val_target_mask = char_mask[split:]
        else:
            train_target_mask = token_target_mask[:split]
            val_target_mask = token_target_mask[split:]
        if int(train_target_mask.sum().item()) == 0:
            raise ValueError(
                "assistant_only_loss enabled but no assistant target tokens were found in train split. "
                "Check dataset format or prefixes."
            )
        print(
            "assistant-only supervision enabled:"
            f" spans={assistant_spans}, "
            f"train_target_tokens={int(train_target_mask.sum().item())}, "
            f"val_target_tokens={int(val_target_mask.sum().item())}"
        )

    train_data = TokenDatasetLite(train_tokens, data_block_size, target_mask=train_target_mask)
    val_data = TokenDatasetLite(val_tokens, data_block_size, target_mask=val_target_mask)

    if resume_checkpoint is not None:
        if "config" not in resume_checkpoint:
            raise ValueError("Checkpoint missing `config`; cannot resume.")
        config = GPTConfig(**resume_checkpoint["config"])
        print("using model config from checkpoint")
    else:
        config = GPTConfig(
            block_size=args.block_size,
            vocab_size=vocab_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
        )

    if vocab_size != config.vocab_size:
        raise ValueError(
            f"Tokenizer vocab_size ({vocab_size}) does not match model vocab_size ({config.vocab_size})."
        )
    model = GPT(config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {num_params:,}")
    print(f"dataset chars: {len(text):,}, tokens: {len(token_ids):,}, vocab: {vocab_size:,}")

    decay_params = [p for p in model.parameters() if p.dim() >= 2]
    nodecay_params = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=args.learning_rate,
        betas=(0.9, 0.95),
    )

    start_step = 0
    best_val_loss = float("inf")
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        if "optimizer_state_dict" in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        start_step = int(resume_checkpoint.get("step", 0))
        best_val_loss = float(resume_checkpoint.get("best_val_loss", float("inf")))
        print(f"loaded checkpoint step: {start_step}")

    if args.best_save_path:
        best_save_path = args.best_save_path
    else:
        root, ext = os.path.splitext(args.save_path)
        best_save_path = f"{root}.best{ext}" if ext else f"{args.save_path}.best"

    if args.lr_decay_steps > 0:
        lr_decay_steps = args.lr_decay_steps
    else:
        lr_decay_steps = start_step + args.steps

    def get_lr(step):
        if args.warmup_steps > 0 and step < args.warmup_steps:
            return args.learning_rate * (step + 1) / args.warmup_steps
        if step >= lr_decay_steps:
            return args.min_lr
        if lr_decay_steps <= args.warmup_steps:
            return args.min_lr
        decay_ratio = (step - args.warmup_steps) / (lr_decay_steps - args.warmup_steps)
        decay_ratio = min(max(decay_ratio, 0.0), 1.0)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return args.min_lr + coeff * (args.learning_rate - args.min_lr)

    def build_checkpoint(step_value, best_val):
        return {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config.__dict__,
            "tokenizer": tokenizer_meta,
            "args": vars(args),
            "step": step_value,
            "best_val_loss": best_val,
            "assistant_only_loss": assistant_only_loss,
        }

    t0 = time.time()
    last_train_loss = None
    for step in range(start_step, start_step + args.steps):
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        x, y, y_mask = train_data.next_batch(
            args.batch_size,
            device,
            require_target=assistant_only_loss,
        )
        _, loss = model(x, y, loss_mask=y_mask)
        last_train_loss = float(loss.item())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % args.eval_interval == 0 or step == (start_step + args.steps - 1):
            losses = estimate_loss(
                model=model,
                train_data=train_data,
                val_data=val_data,
                batch_size=args.batch_size,
                eval_iters=args.eval_iters,
                device=device,
                require_target=assistant_only_loss,
            )
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                if not args.no_checkpoint:
                    best_save_dir = os.path.dirname(best_save_path)
                    if best_save_dir:
                        os.makedirs(best_save_dir, exist_ok=True)
                    torch.save(build_checkpoint(step + 1, best_val_loss), best_save_path)
                best_msg = f" | best_val {best_val_loss:.4f}"
            else:
                best_msg = ""
            elapsed = time.time() - t0
            print(
                f"step {step:5d} | train {losses['train']:.4f} | "
                f"val {losses['val']:.4f} | lr {lr:.2e} | elapsed {elapsed:.1f}s{best_msg}"
            )

    if args.no_checkpoint:
        print("checkpoint saving skipped (--no_checkpoint)")
    else:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        checkpoint = build_checkpoint(start_step + args.steps, best_val_loss)
        torch.save(checkpoint, args.save_path)
        if not os.path.exists(best_save_path):
            best_save_dir = os.path.dirname(best_save_path)
            if best_save_dir:
                os.makedirs(best_save_dir, exist_ok=True)
            torch.save(checkpoint, best_save_path)
        print(f"saved checkpoint to: {args.save_path}")
        print(f"best checkpoint: {best_save_path} (val {best_val_loss:.4f})")

    model.eval()
    prompt_ids = encode(args.start_text)
    if len(prompt_ids) == 0:
        prompt_ids = [0]
    if len(prompt_ids) > config.block_size:
        prompt_ids = prompt_ids[-config.block_size:]
    stop_at_user = args.stop_at_user or conversation_format
    stop_ids = None
    if stop_at_user:
        stop_ids = encode(f"\n{args.user_prefix}")
    x = torch.tensor(prompt_ids, dtype=torch.long, device=device)[None, :]
    with torch.no_grad():
        y = generate(
            model=model,
            idx=x,
            block_size=config.block_size,
            max_new_tokens=args.generate_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            stop_ids=stop_ids,
        )
    print("\n--- sample ---")
    print(decode(y[0].tolist()))
    print("--------------")

    run_summary = {
        "params": num_params,
        "device": device,
        "steps": args.steps,
        "last_train_loss": last_train_loss,
        "checkpoint": None if args.no_checkpoint else args.save_path,
        "best_checkpoint": None if args.no_checkpoint else best_save_path,
        "best_val_loss": best_val_loss,
        "assistant_only_loss": assistant_only_loss,
        "conversation_format": conversation_format,
        "stop_at_user": bool(stop_at_user),
    }
    print("summary:", json.dumps(run_summary))
