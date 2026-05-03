# Mental Health GPT Chatbot

A Streamlit class-project chatbot built around a small GPT-2 style model
fine-tuned on mental-health counseling conversations.

## Run Locally

Install dependencies:

```bash
python -m venv gpt-env
source gpt-env/bin/activate
pip install -r requirements.txt
```

Put the checkpoint at:

```text
models/mental_health_chat/ckpt.pt
```

Start the app:

```bash
streamlit run streamlit_app.py
```

## Deploy

Do not push the model checkpoint to GitHub. The checkpoint is too large for
normal git. Upload the smaller inference checkpoint to Hugging Face Hub and set
these Streamlit secrets:

```toml
HF_MODEL_REPO = "your-huggingface-username/your-model-repo"
HF_MODEL_FILE = "mental_health_infer.pt"
```

If your Hugging Face model repo is private, also add:

```toml
HF_TOKEN = "your_huggingface_read_token"
```

See `DEPLOY_STREAMLIT.md` for the full deployment workflow.

## Project Files

- `streamlit_app.py`: Streamlit chatbot UI.
- `chatbot_inference.py`: model loading and text generation.
- `model.py`: GPT model implementation.
- `prepare_mental_health_data.py`: Hugging Face dataset preparation.
- `train.py`: training script.
- `export_inference_checkpoint.py`: strips optimizer state from a checkpoint.

## Note

This is an educational prototype. It is not a medical product, therapist,
diagnosis tool, or emergency service.
