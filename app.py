import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import os

# Label kelas sesuai urutan direktori
CLASSES = ['closed_look', 'forward_look', 'left_look', 'right_look']

# Load model hanya sekali saat startup
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model/gaze_model.h5')

model = load_model()

def predict_gaze(img: Image.Image):
    img = img.convert('RGB')
    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    class_idx = np.argmax(pred[0])
    confidence = pred[0][class_idx]

    return CLASSES[class_idx], confidence

# UI
st.title("🎯 Deteksi Arah Pandangan Mahasiswa")

uploaded_file = st.file_uploader("📷 Upload gambar wajah mahasiswa", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='🖼️ Gambar yang di-upload', use_column_width=True)

    with st.spinner("🔍 Mendeteksi arah pandangan..."):
        label, confidence = predict_gaze(img)
    
    st.markdown(f"### ✅ Prediksi: **{label.replace('_', ' ').title()}** ({confidence*100:.2f}% yakin)")
