# CIFAR10 Streamlit App

This app loads a trained CIFAR10 CNN and predicts the class of an uploaded image.  
It expects a model file in the project root named `optimized_cnn_model.keras` or `optimized_cnn_model.h5`.

## Local run
1. Put your model file next to `app.py`.
2. `pip install -r requirements.txt`
3. `streamlit run app.py`

## Export model from Colab
If your training ran in Colab and produced `optimized_cnn_model.keras`:
```python
from google.colab import files
files.download("optimized_cnn_model.keras")
