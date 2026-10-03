"""
IMDB Sentiment Analyzer - Streamlit app
Classifies movie reviews as positive or negative using a Simple RNN.

"""

from pathlib import Path

import numpy as np
import streamlit as st
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
MAX_FEATURES = 10000   # vocabulary size used during training
MAX_LEN = 500          # padded sequence length used during training

MODEL_CANDIDATES = [
    Path("models/simple_rnn_model.h5"),
    Path("SimpleRNN/simple_rnn_model.h5"),
    Path("simple_rnn_model.h5"),
    Path(__file__).parent.parent / "models" / "simple_rnn_model.h5",
    Path(__file__).parent / "SimpleRNN" / "simple_rnn_model.h5",
    Path(__file__).parent / "simple_rnn_model.h5",
]

EXAMPLES = {
    "😍 Loved it": "This movie was absolutely fantastic! The acting was superb, the story was gripping, and I was on the edge of my seat the whole time.",
    "😡 Hated it": "What a terrible waste of time. The plot made no sense, the acting was wooden, and I couldn't wait for it to end.",
    "😐 Mixed feelings": "The visuals were beautiful but the story dragged on and the ending felt rushed. It had some good moments, though.",
    "🎬 Critic style": "A beautifully shot and thoughtfully written drama, anchored by a magnetic lead performance that elevates the whole film.",
}

st.set_page_config(
    page_title="IMDB Sentiment Analyzer",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; max-width: 820px; }

    .hero {
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        padding: 2.2rem 1.5rem;
        border-radius: 20px;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(37, 117, 252, 0.35);
        margin-bottom: 1.5rem;
    }
    .hero h1 { margin: 0; font-size: 2.3rem; font-weight: 800; color: white; }
    .hero p  { margin: .5rem 0 0; font-size: 1.05rem; opacity: .92; color: white; }

    .result-card {
        padding: 1.6rem 1.4rem;
        border-radius: 18px;
        text-align: center;
        color: white;
        margin-top: 1rem;
        animation: pop .45s ease;
        box-shadow: 0 8px 24px rgba(0,0,0,.18);
    }
    .positive { background: linear-gradient(135deg, #11998e, #38ef7d); }
    .negative { background: linear-gradient(135deg, #cb2d3e, #ef473a); }
    .result-card .emoji { font-size: 3.2rem; line-height: 1; }
    .result-card .label { font-size: 1.8rem; font-weight: 800; margin-top: .3rem; }
    .result-card .conf  { font-size: 1.05rem; opacity: .95; }

    .meter {
        height: 14px; border-radius: 10px; background: rgba(255,255,255,.3);
        overflow: hidden; margin: .9rem auto 0; max-width: 420px;
    }
    .meter > div { height: 100%; background: white; border-radius: 10px; }

    .history-item {
        padding: .65rem .9rem; border-radius: 12px; margin-bottom: .5rem;
        background: rgba(128,128,128,.10); border-left: 5px solid;
        font-size: .93rem;
    }
    .history-item.positive { border-color: #2ecc71; background: rgba(46,204,113,.10); }
    .history-item.negative { border-color: #e74c3c; background: rgba(231,76,60,.10); }

    div.stButton > button {
        border-radius: 12px; font-weight: 700; padding: .6rem 1rem;
        transition: transform .15s ease, box-shadow .15s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,0,0,.2);
    }
    @keyframes pop {
        from { opacity: 0; transform: scale(.92); }
        to   { opacity: 1; transform: scale(1); }
    }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Model + helpers
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the Simple RNN model…")
def load_assets():
    word_index = imdb.get_word_index()
    model = None
    for path in MODEL_CANDIDATES:
        if path.exists():
            model = load_model(str(path), compile=False)
            break
    return model, word_index


def preprocess(text: str, word_index: dict) -> np.ndarray:
    words = text.lower().split()
    # Keras IMDB reserves indices 0-2, so shift by 3 (2 = unknown token)
    encoded = [min(word_index.get(w, 2) + 3, MAX_FEATURES - 1) for w in words]
    return sequence.pad_sequences([encoded], maxlen=MAX_LEN)


def predict(text: str, model, word_index: dict):
    score = float(model.predict(preprocess(text, word_index), verbose=0)[0][0])
    label = "Positive" if score > 0.5 else "Negative"
    confidence = score if score > 0.5 else 1 - score
    return label, confidence, score


# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
if "review_text" not in st.session_state:
    st.session_state.review_text = ""
if "history" not in st.session_state:
    st.session_state.history = []


def set_example(text: str):
    st.session_state.review_text = text


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎬 About")
    st.write(
        "This app uses a **Simple RNN** with an embedding layer, trained on the "
        "**IMDB 50K movie reviews** dataset, to decide whether a review is "
        "positive or negative."
    )

    st.markdown("### 🧠 Model pipeline")
    st.markdown(
        """
        1. **Tokenize** the review into words
        2. **Encode** words with the IMDB vocabulary
        3. **Pad** to a fixed length of 500
        4. **Embedding → Simple RNN → Sigmoid**
        """
    )

    st.markdown("### ⚙️ Settings")
    show_raw = st.toggle("Show raw model score", value=False)
    max_history = st.slider("History size", 3, 15, 6)

    if st.button("🗑️ Clear history", use_container_width=True):
        st.session_state.history = []
        st.rerun()

    st.divider()
    st.caption("Built with TensorFlow, Keras & Streamlit.")
    st.caption("[GitHub repo](https://github.com/AkshatKumarSingh001/IMDB_sentiment_analyzer)")

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🎬 IMDB Sentiment Analyzer</h1>
        <p>Paste a movie review and let a Simple RNN tell you how it feels.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

model, word_index = load_assets()

if model is None:
    st.error(
        "Trained model file **simple_rnn_model.h5** was not found. "
        "Run `Simplernn.ipynb` to train and save it to `SimpleRNN/simple_rnn_model.h5`, "
        "then restart the app."
    )
    st.stop()

# ----------------------------------------------------------------------------
# Input
# ----------------------------------------------------------------------------
st.markdown("#### ✨ Try an example")
cols = st.columns(len(EXAMPLES))
for col, (name, text) in zip(cols, EXAMPLES.items()):
    col.button(name, on_click=set_example, args=(text,), use_container_width=True)

st.markdown("#### ✍️ Your review")
review = st.text_area(
    "Movie review",
    key="review_text",
    height=170,
    placeholder="e.g. The movie was engaging, well acted, and thoroughly enjoyable.",
    label_visibility="collapsed",
)

word_count = len(review.split())
st.caption(f"📝 {word_count} words · {len(review)} characters")

analyze = st.button("🔍 Analyze sentiment", type="primary", use_container_width=True)

# ----------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------
if analyze:
    if not review.strip():
        st.warning("Please enter a review first.")
    else:
        with st.spinner("Reading between the lines…"):
            label, confidence, raw = predict(review, model, word_index)

        is_pos = label == "Positive"
        css = "positive" if is_pos else "negative"
        emoji = "😊" if is_pos else "😞"

        st.markdown(
            f"""
            <div class="result-card {css}">
                <div class="emoji">{emoji}</div>
                <div class="label">{label} review</div>
                <div class="conf">Confidence: {confidence * 100:.1f}%</div>
                <div class="meter"><div style="width:{confidence * 100:.1f}%"></div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if is_pos:
            st.balloons()

        m1, m2, m3 = st.columns(3)
        m1.metric("Sentiment", label)
        m2.metric("Confidence", f"{confidence * 100:.1f}%")
        m3.metric("Words", word_count)

        if show_raw:
            st.info(f"Raw sigmoid output: **{raw:.4f}** (above 0.5 means positive)")

        if word_count < 5:
            st.caption("💡 Short reviews give less reliable results. Try writing a few more words.")

        st.session_state.history.insert(
            0,
            {"text": review.strip(), "label": label, "conf": confidence, "css": css},
        )
        st.session_state.history = st.session_state.history[:max_history]

# ----------------------------------------------------------------------------
# History
# ----------------------------------------------------------------------------
if st.session_state.history:
    st.markdown("---")
    st.markdown("#### 🕘 Recent analyses")
    for item in st.session_state.history[:max_history]:
        preview = item["text"][:110] + ("…" if len(item["text"]) > 110 else "")
        icon = "👍" if item["css"] == "positive" else "👎"
        st.markdown(
            f"""
            <div class="history-item {item['css']}">
                {icon} <b>{item['label']}</b> ({item['conf'] * 100:.0f}%) — {preview}
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    "<p style='text-align:center; opacity:.6; margin-top:2rem;'>"
    "Made with ❤️ by Akshat Kumar Singh</p>",
    unsafe_allow_html=True,
)