"""
predict.py
High-level predict() function: PIL image + model name → caption + inference time.
"""

import time

from utils.feature_extractor import extract_features
from utils.caption_utils import generate_greedy, generate_beam


def predict(pil_image, model, vocab, idx2word, model_name: str) -> tuple[str, float]:
    """
    pil_image  : PIL.Image
    model      : loaded PyTorch model
    vocab      : dict word→id
    idx2word   : dict id→word
    model_name : "LSTM" | "GRU" | "Transformer"

    Returns (caption_string, inference_seconds)
    """
    t0 = time.perf_counter()

    features = extract_features(pil_image)   # (49, 1280) numpy

    if model_name == "Transformer":
        caption = generate_beam(model, features, vocab, idx2word, beam_width=3)
    else:
        caption = generate_greedy(model, features, vocab, idx2word)

    elapsed = time.perf_counter() - t0
    return caption, elapsed

