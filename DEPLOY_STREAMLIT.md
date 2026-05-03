# Deploying the Streamlit Chatbot

You can put the code on GitHub and deploy the Streamlit app, but do not commit
the trained model checkpoint directly to normal git. The current checkpoint is
about 1.4 GB, and GitHub regular repositories reject files over 100 MB.

## What Goes on GitHub

Commit the app/code files:

```text
streamlit_app.py
chatbot_inference.py
model.py
sample.py
train.py
prepare_mental_health_data.py
export_inference_checkpoint.py
requirements.txt
COLAB_MENTAL_HEALTH.md
DEPLOY_STREAMLIT.md
```

Do not commit:

```text
models/
checkpoints/
data/
out-mental-health-gpt2/
gpt-env/
*.pt
```

These are already covered by `.gitignore`.

## Make a Smaller Inference Checkpoint

The downloaded `best.pt` includes optimizer state for resuming training. The
Streamlit chatbot only needs the model weights.

```bash
./gpt-env/bin/python export_inference_checkpoint.py \
  --src models/mental_health_chat/ckpt.pt \
  --dst mental_health_infer.pt
```

For an even smaller file:

```bash
./gpt-env/bin/python export_inference_checkpoint.py \
  --src models/mental_health_chat/ckpt.pt \
  --dst mental_health_infer_fp16.pt \
  --fp16
```

Use the smaller checkpoint for deployment.

## Recommended Deployment Setup

1. Push the code to GitHub.
2. Upload `mental_health_infer.pt` or `mental_health_infer_fp16.pt` to a Hugging
   Face model repository.
3. Deploy the GitHub repo on Streamlit Community Cloud.
4. In Streamlit app secrets/environment variables, set:

```text
HF_MODEL_REPO=your-huggingface-username/your-model-repo
HF_MODEL_FILE=mental_health_infer.pt
```

If the Hugging Face repo is private, also set:

```text
HF_TOKEN=your_huggingface_read_token
```

The app will download the checkpoint at startup with `hf_hub_download`.

## Local Run

Locally, the app still reads:

```text
models/mental_health_chat/ckpt.pt
```

Run:

```bash
./gpt-env/bin/streamlit run streamlit_app.py
```

## Important Limitation

Streamlit Community Cloud is CPU-based and has memory limits. A GPT-2 checkpoint
can load slowly or exceed memory, especially if you use the full 1.4 GB training
checkpoint. Use the stripped inference checkpoint, and use the fp16 checkpoint if
startup memory is a problem.
