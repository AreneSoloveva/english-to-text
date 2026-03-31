import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import io
import numpy as np
import re

# Hugging Face TrOCR
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

st.set_page_config(page_title="TrOCR + Перевод", page_icon="🔍", layout="wide")

st.title("🔍 **TrOCR OCR + Перевод на русский**")
st.markdown("### Распознавание печатного английского текста с помощью Microsoft TrOCR")

@st.cache_resource
def load_model():
    """Загружаем модель один раз"""
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
    return processor, model

processor, model = load_model()

# Простой словарь переводов (можно заменить на настоящий переводчик позже)
TRANSLATIONS = {
    'hello': 'привет', 'world': 'мир', 'click': 'нажмите', 'open': 'откройте',
    'ok': 'ОК', 'cancel': 'отмена', 'one': 'один', 'two': 'два', 'three': 'три',
    'yes': 'да', 'no': 'нет', 'the': 'the', 'and': 'и', 'is': 'есть'
}

def simple_translate(text):
    words = text.lower().split()
    translated = []
    for word in words:
        clean = re.sub(r'[^\w]', '', word)
        translated.append(TRANSLATIONS.get(clean, word))
    return ' '.join(translated).capitalize()

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
    st.image(image, caption="Оригинал", use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    if st.button("🚀 **Распознать + Перевести**", type="primary"):
        with st.spinner("🔄 TrOCR распознаёт текст... (это может занять 5–15 сек)"):
            processed = preprocess_image(image)
            
            # TrOCR обработка
            pixel_values = processor(images=processed, return_tensors="pt").pixel_values
            generated_ids = model.generate(pixel_values, max_new_tokens=100)
            english_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            english_text = re.sub(r'\s+', ' ', english_text).strip()
        
        with col1:
            st.success("🇬🇧 **Распознанный английский текст:**")
            st.code(english_text, language="text")
        
        with col2:
            russian_text = simple_translate(english_text)
            st.success("🇷🇺 **Перевод на русский:**")
            st.code(russian_text, language="text")

st.info("""
**Используется модель:** microsoft/trocr-base-printed  
✅ Хорошо работает с печатным английским текстом  
⚠️ На Streamlit Cloud работает только CPU-версия (медленнее, чем локально)
""")

st.caption("TrOCR + Streamlit Cloud • 2026")
