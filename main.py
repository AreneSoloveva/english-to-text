import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import io
import pytesseract
import numpy as np
import re

# Конфигурация (Tesseract OCR)
st.set_page_config(page_title="Tesseract OCR", page_icon="🔍", layout="wide")

st.title("🔍 **Tesseract OCR**")
st.markdown("### Распознавание текста с изображений")

# Улучшение изображения
def preprocess_image(image):
    """Предобработка для лучшего OCR"""
    # Конвертация в grayscale
    gray = image.convert('L')
    
    # Увеличение контраста
    enhancer = ImageEnhance.Contrast(gray)
    gray = enhancer.enhance(2.0)
    
    # Размытие для сглаживания
    gray = gray.filter(ImageFilter.MedianFilter())
    
    return gray

# Словарь переводов
TRANSLATIONS = {
    'hello': 'привет', 'world': 'мир', 'click': 'нажмите', 
    'open': 'откройте', 'ok': 'ОК', 'cancel': 'отмена',
    'one': 'один', 'two': 'два', 'three': 'три', 'yes': 'да', 'no': 'нет'
}

def translate_text(text):
    words = text.lower().split()
    translated = []
    for word in words:
        clean = re.sub(r'[^\w]', '', word)
        translated.append(TRANSLATIONS.get(clean, f"[{word}]"))
    return ' '.join(translated).capitalize()

# Загрузка изображения
uploaded_file = st.file_uploader("📁 Изображение", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Оригинал", use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    if st.button("🚀 **OCR + ПЕРЕВОД**", type="primary"):
        with st.spinner("🔄 Tesseract OCR..."):
            # Предобработка
            processed = preprocess_image(image)
            
            # Распознавание
            english_text = pytesseract.image_to_string(processed, lang='eng')
            
            # Очистка
            english_text = re.sub(r'\n+', ' ', english_text).strip()
        
        with col1:
            st.success("🇬🇧 **English:**")
            st.code(english_text)
        
        with col2:
            russian_text = translate_text(english_text)
            st.success("🇷🇺 **Русский:**")
            st.code(russian_text)

# Инструкция
st.info("""
**Tesseract OCR:**
- ✅ Бесплатно и стабильно
- ✅ Работает на Streamlit Cloud
- ✅ Без PyTorch/transformers
- 🎯 Лучше печатный текст
""")

st.caption("✅ Tesseract OCR | Streamlit Cloud 2026")
