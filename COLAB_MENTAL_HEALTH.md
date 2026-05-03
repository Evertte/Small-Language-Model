# Google Colab: Mental Health GPT Chatbot

This workflow uses the Hugging Face dataset already configured in
`prepare_mental_health_data.py`, trains the repo's GPT model in Colab, saves the
checkpoint, then brings it back to VS Code for `streamlit_app.py`.

## 1. Colab Runtime

In Colab:

1. Runtime > Change runtime type.
2. Hardware accelerator: `A100 GPU` if available, otherwise `L4 GPU` or `T4 GPU`.
3. High RAM: on if available.

## 2. Upload Project Files

Run this Colab cell and upload these files from this folder:

```text
configurator.py
model.py
prepare_mental_health_data.py
sample.py
train.py
```

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

## 3. Install Dependencies

```bash
!pip -q install datasets tiktoken transformers
```

## 4. Prepare the Mental Health Dataset

```bash
!python prepare_mental_health_data.py \
  --dataset_id Amod/mental_health_counseling_conversations \
  --out_dir data/mental_health_chat \
  --train_ratio 0.9 \
  --repeat 4
```

Check that the dataset files were created:

```bash
!ls -lh data/mental_health_chat
```

## 5. Smoke Test

Run this first to confirm the code and data work:

```bash
!python train.py \
  --dataset=mental_health_chat \
  --out_dir=out-mental-health-smoke \
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

Use this for the real run:

```bash
!python train.py \
  --dataset=mental_health_chat \
  --out_dir=out-mental-health-gpt2 \
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

If Colab runs out of memory, use the lighter version:

```bash
!python train.py \
  --dataset=mental_health_chat \
  --out_dir=out-mental-health-gpt2 \
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

The training script now saves:

```text
out-mental-health-gpt2/ckpt.pt
out-mental-health-gpt2/best.pt
out-mental-health-gpt2/final.pt
```

## 7. Test in Colab

```bash
!python sample.py \
  --ckpt=out-mental-health-gpt2/best.pt \
  --start="User: I feel anxious and overwhelmed lately.\nAssistant:" \
  --max_new_tokens=220 \
  --temperature=0.7 \
  --top_k=40 \
  --device=cuda
```

## 8. Download the Model

You do not need to download the whole folder. For the Streamlit chatbot, download
one checkpoint file.

Recommended:

```python
from google.colab import files
files.download("out-mental-health-gpt2/best.pt")
```

If you prefer the latest final weights instead of the best validation checkpoint:

```python
from google.colab import files
files.download("out-mental-health-gpt2/final.pt")
```

If the browser still crashes, create a smaller inference-only checkpoint first.
This removes the optimizer state, which is only needed if you want to resume
training:

```python
import os, torch
from google.colab import files

src = "out-mental-health-gpt2/best.pt"
dst = "mental_health_infer.pt"

ckpt = torch.load(src, map_location="cpu")
small = {
    key: ckpt[key]
    for key in ["model", "model_args", "iter_num", "best_val_loss", "config"]
    if key in ckpt
}
torch.save(small, dst)

print(f"{dst}: {os.path.getsize(dst) / (1024 ** 2):.1f} MB")
files.download(dst)
```

Then move the downloaded file into this project as:

```text
models/mental_health_chat/ckpt.pt
```

Alternative: zip the trained model folder only if your browser can handle a
large download:

```bash
!zip -r mental_health_model.zip out-mental-health-gpt2
```

Download it:

```python
from google.colab import files
files.download("mental_health_model.zip")
```

In VS Code, unzip it into this project. The Streamlit app can read it directly
from:

```text
out-mental-health-gpt2/best.pt
```

Or copy your preferred checkpoint to:

```text
models/mental_health_chat/ckpt.pt
```

## 9. Run the Streamlit Chatbot Locally

From this project folder:

```bash
python -m venv gpt-env
source gpt-env/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

If the app does not auto-detect the checkpoint, paste the checkpoint path in the
sidebar, for example:

```text
out-mental-health-gpt2/best.pt
```

## Important Safety Note

This is a learning prototype, not a therapist or medical tool. The app includes
a crisis fallback, but the model can still hallucinate and should not be used
for diagnosis, treatment decisions, or emergencies.
