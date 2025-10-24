import os
import logging
import numpy as np
import streamlit as st

# Keep heavy Keras/TensorFlow imports inside functions if possible
from tensorflow.keras.preprocessing import sequence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = "simple_rnn_imdb.h5"
MAX_LEN = 500  # must match how the model was trained


@st.cache_resource
def get_word_index():
    try:
        from tensorflow.keras.datasets import imdb
        word_index = imdb.get_word_index()
        reverse_word_index = {value: key for key, value in word_index.items()}
        return word_index, reverse_word_index
    except Exception:
        logger.exception("Failed to load IMDB word index.")
        raise


@st.cache_resource
def load_tf_model(path: str = MODEL_PATH):
    """
    Load a Keras model from HDF5 with fallbacks to handle a config mismatch
    where older/newer Keras added 'time_major' or other kwargs that
    the current SimpleRNN does not accept.

    Strategy:
    1) Try normal load_model(..., compile=False).
    2) If ValueError about unrecognized kwargs (e.g. time_major), try loading
       with a custom SimpleRNN subclass that strips 'time_major'.
    3) If that fails, try reconstructing the model from its JSON config
       by removing 'time_major' keys and then loading weights.
    """
    try:
        from tensorflow.keras.models import load_model
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at: {path}")
        model = load_model(path, compile=False)
        logger.info("Model loaded successfully (standard load).")
        return model
    except ValueError as e:
        msg = str(e)
        # handle the specific scenario you reported
        if 'time_major' in msg or 'Unrecognized keyword arguments' in msg:
            logger.warning("ValueError during load_model appears to reference 'time_major'. Trying fallback loader.")
            try:
                # Import tensorflow here
                import tensorflow as tf

                # Custom subclass that ignores time_major kwarg if present in config
                class SimpleRNNNoTimeMajor(tf.keras.layers.SimpleRNN):
                    def __init__(self, *args, **kwargs):
                        kwargs.pop('time_major', None)
                        super().__init__(*args, **kwargs)

                from tensorflow.keras.models import load_model
                model = load_model(path, custom_objects={'SimpleRNN': SimpleRNNNoTimeMajor}, compile=False)
                logger.info("Model loaded successfully using SimpleRNN fallback.")
                return model
            except Exception:
                logger.exception("Fallback using SimpleRNNNoTimeMajor failed; attempting config reconstruction.")

                # Last-resort: load config, strip 'time_major' from JSON, rebuild model and load weights
                try:
                    import h5py
                    import json
                    from tensorflow.keras.models import model_from_json

                    with h5py.File(path, 'r') as f:
                        model_config = f.attrs.get('model_config')
                        if model_config is None:
                            raise RuntimeError("No model_config found in HDF5; cannot reconstruct model.")

                        if isinstance(model_config, bytes):
                            model_config = model_config.decode('utf-8')
                        config = json.loads(model_config)

                        # recursively remove any 'time_major' entries
                        def remove_time_major(obj):
                            if isinstance(obj, dict):
                                obj.pop('time_major', None)
                                for v in obj.values():
                                    remove_time_major(v)
                            elif isinstance(obj, list):
                                for item in obj:
                                    remove_time_major(item)

                        remove_time_major(config)
                        new_config_json = json.dumps(config)
                        model = model_from_json(new_config_json)
                        # load weights from the same file
                        model.load_weights(path)
                        logger.info("Model reconstructed from JSON config and weights (time_major removed).")
                        return model
                except Exception:
                    logger.exception("Config-based reconstruction failed.")
                    raise e  # re-raise original ValueError after attempts
        # If it's some other ValueError, re-raise
        raise


def preprocess_text(text: str, word_index: dict):
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

        preprocessed = preprocess_text(user_input, word_index)
        if preprocessed is None:
            st.warning("Unable to preprocess input. Please try a different review.")
            return

        try:
            with st.spinner("Loading model and running prediction..."):
                model = load_tf_model(MODEL_PATH)
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
