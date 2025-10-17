import io
from PIL import Image
import streamlit as st
from predict_util import load_model, predict_image, CLASS_NAMES

# Konfigurasi halaman
st.set_page_config(page_title="CIFAR10 Classifier", layout="centered")

# Load model hanya sekali (cache)
@st.cache_resource
def _load_model():
    return load_model()

model = _load_model()

# Judul aplikasi
st.title("CIFAR10 Image Classifier")
st.caption("Upload any image. The app resizes to 32x32 like training and predicts the class.")

# Upload gambar
file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if file:
    # Buka gambar
    img = Image.open(io.BytesIO(file.read())).convert("RGB")
    
    # Tampilkan gambar dengan ukuran lebih kecil
    st.image(img, caption="Input", width=250)  # atur ukuran di sini

    # Prediksi top 3 kelas
    topk, probs = predict_image(model, img, top_k=3)

    st.subheader("Top 3 predictions")
    for name, p in topk:
        st.write(f"{name} {p:.4f}")

    # Tampilkan semua probabilitas dalam bentuk bar chart
    st.subheader("All classes")
    st.bar_chart({k: v for k, v in zip(CLASS_NAMES, probs)})

    # Evaluasi manual (opsional)
    st.subheader("Evaluation (optional)")
    true_label = st.text_input("Masukkan label asli (opsional):")
    if true_label:
        result_text = "BENAR" if true_label.lower() == topk[0][0].lower() else "SALAH"
        st.write(f"{result_text} ({topk[0][0]} vs {true_label})")
