import os
import numpy as np
import tensorflow as tf
from PIL import Image

CLASS_NAMES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

def _pick_model_file():
    # prefer Keras v3 format then fallback to H5
    if os.path.exists("optimized_cnn_model.keras"):
        return "optimized_cnn_model.keras"
    if os.path.exists("optimized_cnn_model.h5"):
        return "optimized_cnn_model.h5"
    raise FileNotFoundError("Place optimized_cnn_model.keras or optimized_cnn_model.h5 in the project root")

def load_model(path: str | None = None):
    model_path = path or _pick_model_file()
    ext = os.path.splitext(model_path)[1].lower()
    if ext not in {".keras", ".h5"}:
        raise ValueError("Model file must be .keras or .h5")
    return tf.keras.models.load_model(model_path)

def preprocess_image(img_or_path, target_size=(32, 32)):
    if isinstance(img_or_path, str):
        img = Image.open(img_or_path).convert("RGB")
    else:
        img = img_or_path.convert("RGB")
    img = img.resize(target_size)
    arr = np.asarray(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_image(model, img_or_path, top_k=3):
    x = preprocess_image(img_or_path)
    probs = model.predict(x, verbose=0)[0]
    idxs = np.argsort(probs)[::-1][:top_k]
    topk = [(CLASS_NAMES[i], float(probs[i])) for i in idxs]
    return topk, probs.tolist()
