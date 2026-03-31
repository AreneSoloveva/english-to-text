import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import io
import numpy as np
import re

# Hugging Face TrOCR
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

st.set_page_config(page_title="TrOCR OCR", page_icon="🔍", layout="wide")

st.title("🔍 **TrOCR OCR**")
st.markdown("### Распознавание печатного английского текста с помощью Microsoft TrOCR")

@st.cache_resource
def load_model():
    """Загружаем модель один раз"""
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
    return processor, model

processor, model = load_model()

def preprocess_image(image):
    """Лёгкая предобработка для лучшего качества TrOCR"""
    gray = image.convert('L')
    enhancer = ImageEnhance.Contrast(gray)
    gray = enhancer.enhance(2.0)
    gray = gray.filter(ImageFilter.MedianFilter())
    return gray

# Загрузка изображения
uploaded_file = st.file_uploader("📁 Загрузите изображение с английским текстом", 
                                 type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Оригинал", usecaption("TrOCR + Streamlit Cloud • 2026"))
