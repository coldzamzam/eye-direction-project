import tensorflow as tf
import numpy as np
from keras.preprocessing import image
import gdown
import os

classes = ['closed_look', 'right_look', 'left_look', 'forward_look']

def download_model():
    model_path = 'model/direction_model.h5'
    if not os.path.exists(model_path):
        os.makedirs('model', exist_ok=True)
        url = 'https://drive.google.com/uc?id=1pzCDikwXHML_2OqZe-ZiIJT994aBuUaE'  # ganti dengan File ID kamu
        gdown.download(url, model_path, quiet=False)

download_model()

def predict_gaze(img_path, model_path='model/direction_model.h5'):
    model = tf.keras.models.load_model(model_path)
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    class_idx = np.argmax(pred[0])
    return classes[class_idx], pred[0][class_idx]
