# Google Colab: MSU Data + GPT-2 Fine-Tuning

This notebook workflow builds a Mississippi State University chatbot dataset from official MSU pages, then fine-tunes pretrained GPT-2 with the repo's `train.py` pipeline.

## 1. Pick Runtime

In Colab:

1. Runtime > Change runtime type.
2. Hardware accelerator: `A100 GPU` if available, otherwise `L4 GPU`, otherwise `T4 GPU`.
3. High RAM: on.

## 2. Upload Project Files

Run this cell:

```python
from google.colab import files
import os, shutil

os.makedirs("/content/GPT2", exist_ok=True)
uploaded = files.upload()

for name in uploaded.keys():
    shutil.move(name, f"/content/GPT2/{name}")

%cd /content/GPT2
!ls -la
```

Upload these files:

```text
train.py
model.py
sample.py
configurator.py
prepare_msstate_data.py
```

## 3. Install Dependencies

```bash
!pip -q install requests beautifulsoup4 tiktoken transformers
```

What these do:

- `requests`: downloads official MSU web pages.
- `beautifulsoup4`: extracts readable text from HTML.
- `tiktoken`: tokenizes text with the GPT-2 tokenizer.
- `transformers`: downloads pretrained GPT-2 weights for fine-tuning.

## 4. Build the MSU Dataset

```bash
!python prepare_msstate_data.py \
  --out_dir data/msstate_chat \
  --text_out input.txt \
  --train_ratio 0.9
```

This creates:

```text
input.txt
data/msstate_chat/train.bin
data/msstate_chat/val.bin
data/msstate_chat/meta.pkl
data/msstate_chat/sources.jsonl
data/msstate_chat/failures.jsonl
```

Inspect the dataset:

```bash
!head -n 40 input.txt
!python - <<'PY'
import pickle
with open("data/msstate_chat/meta.pkl", "rb") as f:
    meta = pickle.load(f)
print(meta)
PY
```

## 5. Smoke Test

Run this first. It trains a tiny model for a few iterations to verify the dataset and code work.

```bash
!python train.py \
  --dataset=msstate_chat \
  --out_dir=out-msstate-smoke \
  --device=cuda \
  --init_from=scratch \
  --n_layer=2 \
  --n_head=2 \
  --n_embd=128 \
  --block_size=128 \
  --batch_size=4 \
  --gradient_accumulation_steps=1 \
  --max_iters=20 \
  --eval_interval=10 \
  --eval_iters=5 \
  --compile=False
```

## 6. Fine-Tune GPT-2

This is the real training run. `--init_from=gpt2` loads pretrained GPT-2 weights and continues training on the MSU dataset.

```bash
!python train.py \
  --dataset=msstate_chat \
  --out_dir=out-msstate-gpt2 \
  --device=cuda \
  --init_from=gpt2 \
  --block_size=256 \
  --batch_size=2 \
  --gradient_accumulation_steps=16 \
  --max_iters=3000 \
  --eval_interval=250 \
  --eval_iters=30 \
  --learning_rate=1e-5 \
  --min_lr=1e-6 \
  --warmup_iters=200 \
  --dropout=0.1 \
  --compile=False
```

If Colab runs out of memory, use this lighter version:

```bash
!python train.py \
  --dataset=msstate_chat \
  --out_dir=out-msstate-gpt2 \
  --device=cuda \
  --init_from=gpt2 \
  --block_size=128 \
  --batch_size=1 \
  --gradient_accumulation_steps=16 \
  --max_iters=3000 \
  --eval_interval=250 \
  --eval_iters=30 \
  --learning_rate=1e-5 \
  --min_lr=1e-6 \
  --warmup_iters=200 \
  --dropout=0.1 \
  --compile=False
```

## 7. Test the Checkpoint

```bash
!python sample.py \
  --ckpt=out-msstate-gpt2/ckpt.pt \
  --start="User: How do I apply to Mississippi State as a freshman?\nAssistant:" \
  --max_new_tokens=250 \
  --temperature=0.7 \
  --top_k=40 \
  --device=cuda
```

Try more prompts:

```bash
!python sample.py \
  --ckpt=out-msstate-gpt2/ckpt.pt \
  --start="User: What housing options does Mississippi State offer?\nAssistant:" \
  --max_new_tokens=250 \
  --temperature=0.7 \
  --top_k=40 \
  --device=cuda
```

```bash
!python sample.py \
  --ckpt=out-msstate-gpt2/ckpt.pt \
  --start="User: How much does Mississippi State cost for undergraduate students?\nAssistant:" \
  --max_new_tokens=250 \
  --temperature=0.7 \
  --top_k=40 \
  --device=cuda
```

## 8. What Is Happening

The fine-tuning pipeline is:

```text
official MSU pages
  -> clean text
  -> User/Assistant training examples
  -> GPT-2 token IDs
  -> train.bin and val.bin
  -> pretrained GPT-2
  -> fine-tuned MSU checkpoint
  -> sample.py chatbot test
```

Important: fine-tuning teaches GPT-2 to imitate the MSU dataset, but it can still hallucinate. For a production chatbot, add retrieval over the official source pages so current facts can be looked up at answer time.

## 9. Official Sources Used

`prepare_msstate_data.py` uses official MSU URLs for admissions, scholarships, cost of attendance, academic programs, 2026 academic calendar pages, housing, campus visits, parking permits, and visitor parking. You can add another official MSU page with:

```bash
!python prepare_msstate_data.py \
  --out_dir data/msstate_chat \
  --text_out input.txt \
  --url "https://example.msstate.edu/some-official-page"
```
