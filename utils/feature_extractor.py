"""
utils/feature_extractor.py
Extracts EfficientNetB0 spatial features (49 × 1280) from a PIL image.
Uses TensorFlow/Keras — same pipeline as the training notebooks.
"""

import numpy as np

_model = None   # lazy-loaded once


def _get_model():
    global _model
    if _model is None:
        from tensorflow.keras.applications import EfficientNetB0
        _model = EfficientNetB0(weights="imagenet", include_top=False, pooling=None)
        _model.trainable = False
    return _model


def extract_features(pil_image) -> np.ndarray:
    """
    pil_image : PIL.Image (any size / mode)
    returns   : numpy array (49, 1280)
    """
    from tensorflow.keras.applications.efficientnet import preprocess_input
    from tensorflow.keras.preprocessing.image import img_to_array

    model = _get_model()

    img = pil_image.convert("RGB").resize((224, 224))
    arr = img_to_array(img)
    arr = np.expand_dims(arr, 0)
    arr = preprocess_input(arr)

    feat = model.predict(arr, verbose=0)[0]   # (7, 7, 1280)
    return feat.reshape(-1, 1280)             # (49, 1280)
