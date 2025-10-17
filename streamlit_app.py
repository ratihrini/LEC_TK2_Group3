import io
from PIL import Image
import streamlit as st
from predict_util import load_model, predict_image, CLASS_NAMES

st.set_page_config(page_title="CIFAR10 Classifier", layout="centered")

@st.cache_resource
def _load_model():
    # try .keras first then .h5
    return load_model()

model = _load_model()

st.title("CIFAR10 Image Classifier")
st.caption("Upload any image. The app resizes to 32x32 like training and predicts the class.")

file = st.file_uploader("Upload image", type=["jpg","jpeg","png"])
if file:
    img = Image.open(io.BytesIO(file.read())).convert("RGB")
    st.image(img, caption="Input", use_container_width=True)

    topk, probs = predict_image(model, img, top_k=3)

    st.subheader("Top 3 predictions")
    for name, p in topk:
        st.write(f"{name} {p:.4f}")

    st.subheader("All classes")
    st.bar_chart({k: v for k, v in zip(CLASS_NAMES, probs)})

# 7. Evaluasi sederhana dengan label asli (versi Streamlit)
    st.subheader("Evaluation optional")
    true_label = st.text_input("True label like dog or ship", "")
    if true_label:
        result_text = "BENAR" if true_label.lower() == pred_label.lower() else "SALAH"
        st.write(f"{result_text} ({pred_label} vs {true_label})")

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.text(0.5, 0.5, f"{result_text}\n({pred_label} vs {true_label})",
                ha="center", va="center", fontsize=16)
        ax.axis("off")
        st.pyplot(fig)

