import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import streamlit as st
import predict_util as pu

# Halaman
st.set_page_config(page_title="CIFAR10 Classifier with Unknown Detection", layout="centered")

if st.sidebar.button("Clear cache"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

# Muat model sekali saja
@st.cache_resource
def _load_model():
    return pu.load_model("optimized_cnn_model.keras")

model = _load_model()

# Konstanta
CLASS_NAMES = pu.CLASS_NAMES  # ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
CLASS_SET = set(CLASS_NAMES)

# Normalisasi label manual
def canon_label(label: str) -> str:
    if not label:
        return ""
    label = label.strip().lower()
    synonyms = {
        "aeroplane": "airplane", "plane": "airplane",
        "auto": "automobile", "car": "automobile",
        "boat": "ship", "lorry": "truck",
        "kitty": "cat", "puppy": "dog",
    }
    label = synonyms.get(label, label)
    if label in CLASS_SET:
        return label
    if label in {"unknown", "outside", "oos", "not cifar", "none", "other", "out"}:
        return "outside"
    return "outside"

def evaluate_prediction(pred_label: str, true_label: str) -> str:
    pred_is_out = (pred_label == "unknown")
    true_is_out = (true_label == "outside")
    if pred_is_out and true_is_out:
        return "BENAR. Model menandai di luar CIFAR 10 dan label asli juga di luar."
    if not pred_is_out and not true_is_out and pred_label == true_label:
        return "BENAR. Prediksi cocok dengan label asli."
    if pred_is_out and not true_is_out:
        return "SALAH. Model mengira outside tetapi label asli kelas CIFAR 10."
    if not pred_is_out and true_is_out:
        return "SALAH. Model mengira kelas CIFAR 10 tetapi label asli di luar CIFAR 10."
    return f"SALAH. Prediksi {pred_label}. Asli {true_label}."

# UI
st.title("CIFAR10 Classifier")
st.caption("Upload gambar. Aplikasi akan prediksi kelas dan mendeteksi jika di luar CIFAR 10.")

# Ambang unknown
threshold = st.sidebar.slider("Ambang unknown", min_value=0.10, max_value=0.95, value=0.70, step=0.01)
st.sidebar.write(f"Confidence di bawah {threshold:.2f} dianggap di luar CIFAR 10")

file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(io.BytesIO(file.read())).convert("RGB")
    st.image(img, caption="Gambar Input", width=250)

    # Prediksi
    out = pu.predict_with_threshold(model, img, top_k=3, threshold=threshold)
    max_prob = float(out["confidence"])
    pred_label = "unknown" if out["is_unknown"] else out["label"]
    status = "Kemungkinan di luar CIFAR 10" if max_prob < threshold else "Termasuk CIFAR 10"

    # Plot tiga panel
    fig = plt.figure(figsize=(12, 4))

    # Panel 1: gambar dan ringkasan
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(img)
    ax1.axis("off")
    ax1.set_title(f"{status}\nPrediksi: {pred_label}\nConfidence: {max_prob:.3f}", fontweight="bold")

    # Panel 2: semua kelas
    ax2 = fig.add_subplot(1, 3, 2)
    bars = ax2.bar(CLASS_NAMES, out["probs"])
    ax2.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax2.set_ylabel("Probabilitas")
    ax2.set_title("Probabilitas Semua Kelas")
    ax2.axhline(y=threshold, color="red", linestyle="--", label=f"Threshold {threshold:.2f}")
    ax2.legend()
    if not out["is_unknown"] and out["label"] in CLASS_SET:
        bars[CLASS_NAMES.index(out["label"])].set_color("green")

    # Panel 3: top 3
    ax3 = fig.add_subplot(1, 3, 3)
    top_labels = [k for k, _ in out["topk"]]
    top_probs = [p for _, p in out["topk"]]
    colors = ["green" if p >= threshold else "red" for p in top_probs]
    bars_top = ax3.barh(top_labels, top_probs, color=colors)
    ax3.set_xlabel("Probabilitas")
    ax3.set_title("Top 3 Predictions")
    ax3.set_xlim(0, 1)
    for bar, p in zip(bars_top, top_probs):
        ax3.text(p + 0.01, bar.get_y() + bar.get_height() / 2, f"{p:.3f}", va="center")

    plt.tight_layout()
    st.pyplot(fig)

    # Ringkasan teks
    st.markdown("---")
    st.write(f"Hasil: {status}")
    st.write(f"Prediksi: {pred_label}")
    st.write(f"Confidence: {max_prob:.3f}")
    st.write("Top 3:")
    for i, (lbl, pr) in enumerate(out["topk"], 1):
        tag = "di atas threshold" if pr >= threshold else "di bawah threshold"
        st.write(f"{i}. {lbl}: {pr:.4f} ({tag})")

    # Input label asli dan evaluasi
    true_raw = st.text_input("Masukkan label asli. Contoh frog cat dog. Jika bukan kelas CIFAR 10 tulis apa pun misalnya panda")
    if true_raw:
        true_label = canon_label(true_raw)
        verdict = evaluate_prediction(pred_label, true_label)
        st.write(f"Evaluasi: {verdict}")
else:
    st.info("Pilih file gambar untuk mulai memprediksi.")
