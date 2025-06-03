import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import os
import gdown

# Kelas prediksi
CLASSES = ['closed_look', 'forward_look', 'left_look', 'right_look']
device = torch.device("cpu")

# Define arsitektur CNN
class GazeCNN(nn.Module):
    def __init__(self):
        super(GazeCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 37 * 37, 128), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# Fungsi load model dengan auto-download jika tidak ada
@st.cache_resource
def load_model():
    model_path = 'model/direction_model.pt'
    if not os.path.exists(model_path):
        os.makedirs('model', exist_ok=True)
        gdown.download('https://drive.google.com/uc?id=1RmU1ydOj6K5F1uM-LTblue3gVcqsx59o', model_path, quiet=False)

    model = GazeCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

model = load_model()

# Transformasi gambar
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Fungsi prediksi
def predict_gaze(img: Image.Image):
    img = img.convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        class_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][class_idx].item()
    return CLASSES[class_idx], confidence

# UI Streamlit
st.title("🎯 Deteksi Arah Pandangan Mahasiswa (PyTorch)")

uploaded_file = st.file_uploader("📷 Upload gambar wajah mahasiswa", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='🖼️ Gambar yang di-upload', use_column_width=True)

    with st.spinner("🔍 Mendeteksi arah pandangan..."):
        label, confidence = predict_gaze(img)

    st.markdown(f"### ✅ Prediksi: **{label.replace('_', ' ').title()}** ({confidence*100:.2f}% yakin)")
