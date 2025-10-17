import os
import numpy as np
from PIL import Image
import tensorflow as tf

CLASS_NAMES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
_NUM_CLASSES = len(CLASS_NAMES)

def _is_supported_model_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in {".keras", ".h5"}

def load_model(path: str = "optimized_cnn_model.keras"):
    if not os.path.exists(path):
        alt = "optimized_cnn_model.h5"
        if os.path.exists(alt):
            path = alt
        else:
            raise FileNotFoundError(
                "Model tidak ditemukan. Harus ada optimized_cnn_model.keras atau optimized_cnn_model.h5"
            )
    if not _is_supported_model_file(path):
        raise ValueError("File model harus .keras atau .h5")
    return tf.keras.models.load_model(path)

def _letterbox(img: Image.Image, target_size=(32, 32), bg_color=(0, 0, 0)) -> Image.Image:
    resample = Image.Resampling.BICUBIC
    img = img.copy().convert("RGB")
    img.thumbnail(target_size, resample)
    bg = Image.new("RGB", target_size, bg_color)
    x = (target_size[0] - img.size[0]) // 2
    y = (target_size[1] - img.size[1]) // 2
    bg.paste(img, (x, y))
    return bg

def preprocess_image(img_or_path, target_size=(32, 32), letterbox=True):
    img = Image.open(img_or_path).convert("RGB") if isinstance(img_or_path, str) else img_or_path.convert("RGB")
    if letterbox:
        img = _letterbox(img, target_size)
    else:
        img = img.resize(target_size, Image.Resampling.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, 0)  # (1, H, W, 3)
    return arr

def _softmax_if_needed(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32, copy=False)
    s = float(np.sum(v))
    if s <= 0.0 or not np.isfinite(s) or not np.isclose(s, 1.0, atol=1e-3):
        v = tf.nn.softmax(v).numpy()
    return v

def predict_image(model, img_or_path, top_k=3, letterbox=True):
    x = preprocess_image(img_or_path, letterbox=letterbox)
    probs = model.predict(x, verbose=0).squeeze()
    probs = _softmax_if_needed(probs)
    if probs.shape[0] != _NUM_CLASSES:
        raise ValueError(f"Output model {probs.shape[0]} tidak sama dengan jumlah kelas {_NUM_CLASSES}.")
    k = int(min(top_k, _NUM_CLASSES))
    idxs = np.argsort(probs)[::-1][:k]
    topk = [(CLASS_NAMES[i], float(probs[i])) for i in idxs]
    return topk, probs.tolist()

def _entropy(p: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1.0)
    return float(-np.sum(p * np.log(p)))

def predict_with_threshold(model, img_or_path, top_k=3, threshold=0.5, unknown_label="unknown", letterbox=True):
    topk, probs = predict_image(model, img_or_path, top_k=top_k, letterbox=letterbox)
    pred_label, pred_prob = topk[0]
    is_unknown = float(pred_prob) < float(threshold)
    final_label = unknown_label if is_unknown else pred_label
    entropy = _entropy(np.array(probs, dtype=np.float32))
    return {
        "label": final_label,
        "confidence": float(pred_prob),
        "entropy": entropy,
        "is_unknown": is_unknown,
        "topk": topk,
        "probs": probs,
    }
