import streamlit as st
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import io
import numpy as np

# Конфигурация страницы
st.set_page_config(
    page_title="OCR + Перевод",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 **OCR + Перевод**")
st.markdown("### 🇬🇧 Английский текст → 🇷🇺 Русский (автоперевод)")

# Загрузка модели (кэшируется)
@st.cache_resource
def load_ocr_model():
    """Загрузка легкой TrOCR модели для CPU"""
    model_name = "microsoft/trocr-base-printed"
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    return processor, model

# Загрузка моделей с прогрессом
progress_bar = st.progress(0)
status_text = st.empty()

status_text.text("📥 Загрузка OCR модели...")
processor, model = load_ocr_model()
progress_bar.progress(100)
status_text.success("✅ Модель готова!")

# Сайдбар с инфо
with st.sidebar:
    st.info("**Модель:** microsoft/trocr-base-printed")
    st.info("**CPU оптимизация:** ✅")
    st.caption("Загрузка ~30 сек")

# Загрузка изображения
uploaded_file = st.file_uploader(
    "📁 **Загрузите изображение с английским текстом**",
    type=['png', 'jpg', 'jpeg', 'webp'],
    help="Поддерживает PNG, JPG, WebP"
)

if uploaded_file is not None:
    # Показываем изображение
    image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
    st.image(image, caption="Загруженное изображение", use_column_width=True)
    
    # Кнопка распознавания
    if st.button("🚀 **РАСПОЗНАТЬ ТЕКСТ**", type="primary"):
        with st.spinner("🔄 Распознавание текста..."):
            # Подготовка изображения
            pixel_values = processor(image, return_tensors="pt").pixel_values
            
            # Генерация текста (CPU)
            generated_ids = model.generate(
                pixel_values, 
                max_length=64,
                num_beams=4,
                do_sample=False
            )
            
            # Декодирование
            english_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Результаты
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.success("🇬🇧 **Распознанный текст:**")
            st.code(english_text, language="")
        
        with col2:
            # Простой словарь для демонстрации перевода
            translations = {
                "hello": "привет",
                "world": "мир", 
                "image": "изображение",
                "text": "текст",
                "welcome": "добро пожаловать"
            }
            
            # Автоперевод ключевых слов
            russian_words = []
            english_words = english_text.lower().split()
            
            for word in english_words:
                clean_word = word.strip(".,!?:;").lower()
                if clean_word in translations:
                    russian_words.append(translations[clean_word])
                else:
                    russian_words.append(f"[{word}]")
            
            russian_text = " ".join(russian_words)
            
            st.info("🇷🇺 **Перевод (демо):**")
            st.code(russian_text, language="")

# Инструкция
with st.expander("📖 Как использовать"):
    st.markdown("""
    1. **Загрузите изображение** с четким английским текстом
    2. Нажмите **"РАСПОЗНАТЬ ТЕКСТ"**
    3. Получите **английский текст + базовый перевод**!
    
    💡 **Совет:** Используйте изображения с печатным текстом
    """)

st.markdown("---")
st.caption("🤖 TrOCR-base | Streamlit Cloud 2026")
