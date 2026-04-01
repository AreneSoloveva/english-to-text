import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import re
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

# Кэшируем модель (загружается один раз)
@st.cache_resource
def load_trocr_model():
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
    return processor, model

processor, model = load_trocr_model()

st.set_page_config(page_title="TrOCR OCR", page_icon="🔍", layout="wide")

st.title("🔍 **TrOCR OCR**")
st.markdown("### Распознавание печатного текста (microsoft/trocr-base-printed)")

# Предобработка изображения (оптимально для TrOCR)
def preprocess_image(image):
    """Увеличение контраста и сглаживание"""
    enhancer = ImageEnhance.Contrast(image)
    enhanced = enhancer.enhance(2.0)
    enhanced = enhanced.filter(ImageFilter.MedianFilter())
    return enhanced

# Простой словарь переводов (как в вашем оригинале)
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
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Оригинал", use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    if st.button("🚀 **OCR + ПЕРЕВОД**", type="primary"):
        with st.spinner("🔄 TrOCR распознаёт текст..."):
            processed = preprocess_image(image)
            
            # TrOCR inference
            pixel_values = processor(images=processed, return_tensors="pt").pixel_values
            
            with torch.no_grad():
                generated_ids = model.generate(pixel_values)
            
            english_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Очистка текста
            english_text = re.sub(r'\n+', ' ', english_text).strip()
        
        with col1:
            st.success("🇬🇧 **English:**")
            st.code(english_text)
        
        with col2:
            russian_text = translate_text(english_text)
            st.success("🇷🇺 **Русский:**")
            st.code(russian_text)

st.info("""
**TrOCR (microsoft/trocr-base-printed):**
- ✅ Обязательная модель по вашему условию
- ✅ Лучше всего работает с печатным текстом
- ✅ Transformer-based OCR (без Tesseract)
- ✅ Работает на Streamlit Cloud (CPU)
""")

st.caption("✅ TrOCR + Streamlit Cloud 2026")
