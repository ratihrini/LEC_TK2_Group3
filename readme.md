# CIFAR10 Streamlit App

This app loads a trained CIFAR10 CNN and predicts the class of an uploaded image.  
It expects a model file in the project root named `optimized_cnn_model.keras` or `optimized_cnn_model.h5`.

##Kelompok 3 kelas JPCA
1. Muhammad Ghoni Khidir Tohir	(2802518591) 
2. Ratih Octavia Rini			      (2802552074) 
3. Hella Dwi Pratiwi			      (2802519650) 
4. Ahmad Fauzi				          (2802556955) 
5. Luqman Aulia Gani			      (2802518780)
6. 
## Local run
1. Put your model file next to `app.py`.
2. `pip install -r requirements.txt`
3. `streamlit run app.py`

## Export model from Colab
If your training ran in Colab and produced `optimized_cnn_model.keras`:
```python
from google.colab import files
files.download("optimized_cnn_model.keras")


