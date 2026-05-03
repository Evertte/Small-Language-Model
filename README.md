# Mental Health GPT Chatbot

This project is a small language model chatbot built for a class project. It
uses a GPT-2 style transformer architecture, prepares a mental-health
conversation dataset from Hugging Face, fine-tunes from pretrained GPT-2
weights, and serves the final model through a Streamlit chat interface.

The goal was not only to make a chatbot, but to understand the full workflow:
data preparation, tokenization, transformer training, checkpointing, inference,
web app integration, GitHub deployment, and Hugging Face model hosting.

## End-to-End Flow

```text
Hugging Face dataset
        |
        v
prepare_mental_health_data.py
        |
        v
GPT-2 tokenizer
        |
        v
train.bin / val.bin / meta.pkl
        |
        v
model.py + train.py
        |
        v
GPT-2 initialized fine-tuned checkpoint
        |
        v
Hugging Face model repository
        |
        v
Streamlit chatbot app
```

## What This Project Includes

- A GPT-2 style transformer implementation in `model.py`.
- A full training loop in `train.py`.
- A Hugging Face dataset preparation pipeline in `prepare_mental_health_data.py`.
- A command-line sampling script in `sample.py`.
- A reusable inference module in `chatbot_inference.py`.
- A Streamlit chatbot UI in `streamlit_app.py`.
- Colab training instructions in `COLAB_MENTAL_HEALTH.md`.
- Streamlit and Hugging Face deployment notes in `DEPLOY_STREAMLIT.md`.

## Dataset

The training data comes from:

```text
Amod/mental_health_counseling_conversations
```

The raw dataset contains counseling-style context and response pairs. The
preparation script cleans each example and turns it into a simple chat format:

```text
User: <context>
Assistant: <response>
```

Then it uses the GPT-2 tokenizer through `tiktoken` and saves token IDs into:

```text
data/mental_health_chat/train.bin
data/mental_health_chat/val.bin
data/mental_health_chat/meta.pkl
```

The training script reads these binary files with NumPy memory maps, which is
efficient and matches the nanoGPT-style training workflow.

## Model Architecture

The model follows the GPT-2 decoder-only transformer design:

- token embeddings
- positional embeddings
- stacked transformer blocks
- causal self-attention
- MLP feed-forward layers
- layer normalization
- dropout
- tied input embedding and output language-model head weights
- autoregressive next-token prediction

The model learns to predict the next token in a sequence. During chatbot use, it
receives a prompt ending in:

```text
Assistant:
```

and generates the assistant response token by token.

## Training Strategy

We used two stages while building the project.

First, we ran small smoke-test models from scratch. These runs used tiny model
settings so we could quickly verify that:

- the dataset files were created correctly
- the training loop worked
- checkpoints were saved
- sampling from a checkpoint worked
- the app could load a trained checkpoint

After the pipeline worked, we fine-tuned from pretrained GPT-2 weights:

```bash
--init_from=gpt2
```

This is important because the final model did not start from random weights. It
started with GPT-2's pretrained language ability, then adapted to the
mental-health conversation dataset.

Example Colab fine-tuning command:

```bash
python train.py \
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

The training script saves checkpoints such as:

```text
ckpt.pt
best.pt
final.pt
```

## Streamlit Chatbot

The chatbot UI is built with Streamlit. It supports:

- checkpoint loading
- CPU, MPS, or CUDA device selection
- adjustable response length
- temperature sampling
- top-k sampling
- repetition penalty
- chat history formatting
- token-by-token streaming output

Instead of waiting for the full answer, the app now streams tokens as they are
predicted, giving a typewriter-style chatbot experience.

## Local Setup

Create a virtual environment:

```bash
python -m venv gpt-env
source gpt-env/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Place the trained checkpoint here:

```text
models/mental_health_chat/ckpt.pt
```

Run the app:

```bash
streamlit run streamlit_app.py
```

Open:

```text
http://localhost:8501
```

## Deployment

The source code is hosted on GitHub:

```text
https://github.com/Evertte/Small-Language-Model
```

The model checkpoint is hosted separately on Hugging Face because it is too
large for normal GitHub commits:

```text
https://huggingface.co/Evertte/mental-health-chatbot-model
```

Streamlit Cloud should be configured with:

```text
Repository: Evertte/Small-Language-Model
Branch: main
Main file path: streamlit_app.py
```

Secrets:

```toml
HF_MODEL_REPO = "Evertte/mental-health-chatbot-model"
HF_MODEL_FILE = "ckpt.pt"
```

When the deployed app starts, it downloads the checkpoint from Hugging Face Hub
and loads it into the Streamlit chatbot.

## Important Files

```text
model.py                         GPT-2 style architecture
train.py                         training loop and checkpoint saving
prepare_mental_health_data.py    Hugging Face dataset preparation
sample.py                        checkpoint sampling script
chatbot_inference.py             model loading and token generation
streamlit_app.py                 Streamlit chatbot UI
export_inference_checkpoint.py   optional checkpoint size reducer
COLAB_MENTAL_HEALTH.md           Google Colab training guide
DEPLOY_STREAMLIT.md              deployment guide
requirements.txt                 dependencies
```

## What We Learned

This project demonstrates:

- how transformer language models are structured
- how GPT-style tokenization works
- how text is converted into training binaries
- how next-token prediction training works
- why pretrained GPT-2 initialization improves training
- how sampling settings affect chatbot quality
- how to save, load, and test checkpoints
- how to separate code from large model artifacts
- how to deploy an ML app with GitHub, Hugging Face, and Streamlit

## Limitations

This is an educational prototype. It is not a therapist, medical product,
diagnostic system, or emergency service.

Because it is a small GPT-2 fine-tune, it can still:

- repeat itself
- hallucinate
- give generic responses
- misunderstand user intent
- produce inconsistent answers

Future improvements could include LoRA fine-tuning an instruction model,
training with assistant-only loss, improving the dataset quality, or adding
retrieval over trusted mental-health resources.
