import os
import logging
import numpy as np
import streamlit as st

# TensorFlow / Keras imports kept inside functions where possible to reduce import-time cost
from tensorflow.keras.preprocessing import sequence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = "simple_rnn_imdb.h5"
MAX_LEN = 500  # must match how the model was trained


@st.cache_resource
def get_word_index():
    """
    Lazily fetch the IMDb word index (cached). Doing this at import time can block startup.
    """
    try:
        # import inside function to avoid heavy library work at import-time
        from tensorflow.keras.datasets import imdb
        word_index = imdb.get_word_index()
        # shift indices by 3 in your pipeline, so we keep original mapping here
        reverse_word_index = {value: key for key, value in word_index.items()}
        return word_index, reverse_word_index
    except Exception:
        logger.exception("Failed to load IMDB word index.")
        raise


@st.cache_resource
def load_tf_model(path: str = MODEL_PATH):
    """
    Lazily load the Keras model and cache it for reuse by Streamlit.
    load_model(..., compile=False) can be faster if you don't need to recompile.
    """
    try:
        from tensorflow.keras.models import load_model
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at: {path}")
        # compile=False avoids spend on re-compiling optimizer during load
        model = load_model(path, compile=False)
        logger.info("Model loaded successfully.")
        return model
    except Exception:
        logger.exception("Failed to load the model.")
        raise


def preprocess_text(text: str, word_index: dict):
    """
    Convert plain text to the integer encoded/padded sequence expected by the model.
    Keeps the same +3 offset used in the original decoding pipeline.
    """
    if not text:
        return None
    words = text.lower().split()
    encoded = [word_index.get(word, 2) + 3 for word in words]  # 2 = unknown token
    padded = sequence.pad_sequences([encoded], maxlen=MAX_LEN)
    return padded


def main():
    st.title("NeuroCast")
    st.write("IMDB Movie Review Sentiment Analysis")
    st.write("Enter a movie review to classify it as positive or negative.")

    # Load small resources lazily under a spinner so the app responds quickly
    try:
        with st.spinner("Loading vocabulary..."):
            word_index, reverse_word_index = get_word_index()
    except Exception as e:
        st.error("Could not load IMDB word index. Check logs.")
        st.exception(e)
        return

    # Do not load the TF model until user presses Classify to minimize memory usage at startup.
    user_input = st.text_area("Movie Review", height=150)

    if st.button("Classify"):
        if not user_input or not user_input.strip():
            st.warning("Please enter a movie review before classifying.")
            return

        preprocessed = preprocess_text(user_input, word_index)
        if preprocessed is None:
            st.warning("Unable to preprocess input. Please try a different review.")
            return

        # Load model lazily — cache_resource ensures this happens only once per session/process
        try:
            with st.spinner("Loading model and running prediction..."):
                model = load_tf_model(MODEL_PATH)
                # Use model.predict with verbose=0; wrap in try/except to surfacing exceptions to logs
                preds = model.predict(preprocessed, verbose=0)
        except FileNotFoundError as fnf:
            st.error(f"Model file missing: {fnf}")
            st.info("Place your `simple_rnn_imdb.h5` file in the repo root (or adjust MODEL_PATH).")
            logger.exception("Model file missing.")
            return
        except Exception as e:
            st.error("A problem occurred during prediction. See logs for details.")
            st.exception(e)
            logger.exception("Prediction failed.")
            return

        # Ensure preds shape matches expectation
        try:
            score = float(preds[0][0])
            sentiment = "Positive" if score > 0.5 else "Negative"
            st.write(f"Sentiment: {sentiment}")
            st.write(f"Prediction Score: {score:.4f}")
        except Exception:
            st.error("Unexpected prediction output shape.")
            logger.exception("Unexpected prediction output: %s", preds)
            st.write(preds)
    else:
        st.info("Please enter a movie review and click Classify.")


if __name__ == "__main__":
    main()
