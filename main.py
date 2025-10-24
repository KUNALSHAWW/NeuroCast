import os
import logging
import numpy as np
import streamlit as st

# Keep heavy Keras/TensorFlow imports inside functions if possible
from tensorflow.keras.preprocessing import sequence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = "simple_rnn_imdb.h5"
MAX_LEN = 500          # must match how the model was trained
NUM_WORDS = 10000      # top-N words used when the model was trained (adjust to match your training)


@st.cache_resource
def get_word_index(num_words: int = NUM_WORDS):
    """
    Build a safe word->index mapping consistent with how Keras IMDB data is typically prepared.

    We:
    - load the base mapping from imdb.get_word_index()
    - shift indices by +3 (so reserved tokens 0..3 remain)
    - keep only indices that fall below num_words to avoid out-of-range indices
    - add reserved tokens explicitly
    """
    try:
        from tensorflow.keras.datasets import imdb
        base = imdb.get_word_index()
        # Keep only top `num_words` and shift by +3 to match Keras convention
        filtered = {}
        for word, idx in base.items():
            shifted = idx + 3
            if shifted < num_words:
                filtered[word] = shifted
        # Add reserved tokens
        filtered["<PAD>"] = 0
        filtered["<START>"] = 1
        filtered["<UNK>"] = 2
        filtered["<UNUSED>"] = 3
        reverse = {v: k for k, v in filtered.items()}
        return filtered, reverse
    except Exception:
        logger.exception("Failed to load IMDB word index.")
        raise


@st.cache_resource
def load_tf_model(path: str = MODEL_PATH):
    """
    Load Keras model and log embedding input_dim for diagnostics.
    """
    try:
        from tensorflow.keras.models import load_model
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at: {path}")
        model = load_model(path, compile=False)
        # Log embedding configuration (helpful to detect mismatches)
        try:
            emb = next((layer for layer in model.layers if layer.__class__.__name__ == 'Embedding'), None)
            if emb is not None:
                logger.info(f"Loaded model. Embedding input_dim={getattr(emb, 'input_dim', 'unknown')}, output_dim={getattr(emb, 'output_dim', 'unknown')}")
        except Exception:
            logger.exception("Failed to log embedding info.")
        logger.info("Model loaded successfully.")
        return model
    except Exception:
        logger.exception("Failed to load the model.")
        raise


def preprocess_text(text: str, word_index: dict, num_words: int = NUM_WORDS):
    """
    Convert plain text to integer encoded/padded sequence expected by the model.

    Guarantees:
    - indices are clipped so none exceed num_words-1
    - returns dtype int32 to match embedding requirements
    """
    if not text:
        return None
    words = text.lower().split()
    encoded = []
    for w in words:
        idx = word_index.get(w, 2)  # default to <UNK> index (2)
        # If idx is out of training vocabulary bounds, map to unknown token
        if not isinstance(idx, int) or idx >= num_words or idx < 0:
            idx = 2
        encoded.append(idx)
    padded = sequence.pad_sequences([encoded], maxlen=MAX_LEN)
    return np.array(padded, dtype=np.int32)


def main():
    st.title("NeuroCast")
    st.write("IMDB Movie Review Sentiment Analysis")
    st.write("Enter a movie review to classify it as positive or negative.")

    try:
        with st.spinner("Loading vocabulary..."):
            word_index, reverse_word_index = get_word_index()
    except Exception as e:
        st.error("Could not load IMDB word index. Check logs.")
        st.exception(e)
        return

    user_input = st.text_area("Movie Review", height=150)

    if st.button("Classify"):
        if not user_input or not user_input.strip():
            st.warning("Please enter a movie review before classifying.")
            return

        preprocessed = preprocess_text(user_input, word_index, NUM_WORDS)
        if preprocessed is None:
            st.warning("Unable to preprocess input. Please try a different review.")
            return

        try:
            with st.spinner("Loading model and running prediction..."):
                model = load_tf_model(MODEL_PATH)

                # Diagnostic: check embedding input_dim vs max token index in the input
                emb = next((layer for layer in model.layers if layer.__class__.__name__ == 'Embedding'), None)
                max_token = int(np.max(preprocessed))
                logger.info(f"Max token index in input: {max_token}")
                if emb is not None and hasattr(emb, 'input_dim'):
                    allowed = emb.input_dim
                    logger.info(f"Model embedding allows indices in [0, {allowed - 1}]")
                    if max_token >= allowed:
                        st.error(
                            f"Token index {max_token} is >= embedding input_dim ({allowed}). "
                            "This indicates your preprocessing produced indices outside the vocabulary size used during training. "
                            "Make sure NUM_WORDS matches the number used when training the model."
                        )
                        return

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
