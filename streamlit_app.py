from pathlib import Path
import os

import streamlit as st

from chatbot_inference import generate_reply, load_chat_model


DEFAULT_CKPT = Path("models/mental_health_chat/ckpt.pt")
FALLBACK_CKPTS = [
    DEFAULT_CKPT,
    Path("out-mental-health-gpt2/ckpt.pt"),
    Path("out-mental-health-gpt2/best.pt"),
    Path("out-mental-health-gpt2/final.pt"),
    Path("checkpoints/mental_health_chat.pt"),
]


def download_hf_checkpoint() -> str | None:
    repo_id = get_setting("HF_MODEL_REPO")
    if not repo_id:
        return None

    filename = get_setting("HF_MODEL_FILE", "mental_health_infer.pt")
    token = get_setting("HF_TOKEN") or None
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename=filename, token=token)


def get_setting(name: str, default: str | None = None) -> str | None:
    if os.environ.get(name):
        return os.environ[name]
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


def first_existing_checkpoint() -> str:
    configured_path = get_setting("MODEL_PATH")
    if configured_path and Path(configured_path).exists():
        return configured_path

    for path in FALLBACK_CKPTS:
        if path.exists():
            return str(path)

    hf_path = download_hf_checkpoint()
    if hf_path:
        return hf_path

    return str(DEFAULT_CKPT)


@st.cache_resource(show_spinner="Loading model...")
def cached_model(ckpt_path: str, device: str):
    return load_chat_model(ckpt_path, device=device)


st.set_page_config(page_title="Mental Health GPT Chatbot", layout="centered")

st.title("Mental Health GPT Chatbot")
st.caption(
    "Prototype only. This chatbot is not a therapist, doctor, diagnosis tool, "
    "or emergency service."
)

with st.sidebar:
    st.header("Model")
    ckpt_path = st.text_input("Checkpoint path", value=first_existing_checkpoint())
    device = st.selectbox("Device", ["auto", "cpu", "mps", "cuda"], index=0)
    max_new_tokens = st.slider("Max response tokens", 40, 400, 140, step=20)
    temperature = st.slider("Temperature", 0.1, 1.3, 0.45, step=0.05)
    top_k = st.slider("Top-k", 1, 100, 25, step=1)
    repetition_penalty = st.slider("Repetition penalty", 1.0, 1.5, 1.15, step=0.05)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi. What would you like to talk about today?",
        }
    ]

if st.button("Clear chat"):
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi. What would you like to talk about today?",
        }
    ]
    st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

prompt = st.chat_input("Type a message")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    ckpt = Path(ckpt_path)
    if not ckpt.exists():
        st.error(f"Checkpoint not found: {ckpt}")
        st.stop()

    with st.chat_message("assistant"):
        with st.spinner("Generating..."):
            model, enc, resolved_device = cached_model(str(ckpt), device)
            reply = generate_reply(
                model=model,
                enc=enc,
                device=resolved_device,
                history=st.session_state.messages[:-1],
                user_message=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )
        st.write(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
