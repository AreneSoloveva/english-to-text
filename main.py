import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import re
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
from deep_translator import GoogleTranslator

# Кэшируем модель (загружается один раз при запуске)
@st.cache_resource
def load_trocr_model():
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
    return processor, model

# ДОБАВЛЕНО: кэширование переводчика
@st.cache_resource
def get_translator():
    return GoogleTranslator(source='en', target='ru')

# Загружаем модель
processor, model = load_trocr_model()
translator = get_translator()

# ДОБАВЛЕНО: функция перевода текста с английского на русский
def translate_text(text):
    if not text or not text.strip():
        return ""
    try:
        # При длинном тексте разбиваем на предложения (ограничение Google Translate)
        if len(text) > 4500:
            sentences = re.split(r'(?<=[.!?])\s+', text)
            translated_parts = [translator.translate(s) for s in sentences if s.strip()]
            return ' '.join(translated_parts)
        else:
            return translator.translate(text)
    except Exception as e:
        return f"Ошибка перевода: {e}"

# Оригинальная функция предобработки изображения (не тронута)
def preprocess_image(image):
    enhancer = ImageEnhance.Contrast(image)
    enhanced = enhancer.enhance(2.0)
    enhanced = enhanced.filter(ImageFilter.MedianFilter())
    return enhanced

# ДОБАВЛЕНО: инициализация переменных состояния для хранения распознанного текста, перевода и обработанного изображения
if 'english_text' not in st.session_state:
    st.session_state.english_text = None
if 'russian_text' not in st.session_state:
    st.session_state.russian_text = None
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None

# Интерфейс
st.set_page_config(
    page_title="TrOCR OCR",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 **TrOCR OCR**")
st.markdown("### Распознавание печатного английского текста\n**Модель:** `microsoft/trocr-base-printed`")


# Загрузка изображения
uploaded_file = st.file_uploader(
    "📁 Загрузите изображение с печатным текстом",
    type=['png', 'jpg', 'jpeg', 'tiff']
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    # Показываем оригинал
    st.image(image, caption="📸 Оригинальное изображение", use_container_width=True)

    if st.button("🚀 **Распознать текст (OCR)**", type="primary"):
        with st.spinner("🔄 TrOCR распознаёт текст..."):
            # Предобработка
            processed_image = preprocess_image(image)
            
            # Подготовка для модели
            pixel_values = processor(
                images=processed_image,
                return_tensors="pt"
            ).pixel_values

            # Инференс
            with torch.no_grad():
                generated_ids = model.generate(
                    pixel_values,
                    max_new_tokens=512  # на случай длинного текста
                )

            # Декодирование
            english_text = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]

            # Базовая очистка текста
            english_text = re.sub(r'\s+', ' ', english_text).strip()

        # ДОБАВЛЕНО: сохраняем результаты в session_state вместо прямого вывода
        st.session_state.english_text = english_text
        st.session_state.processed_image = processed_image
        st.session_state.russian_text = None  # сбрасываем перевод
        st.rerun() # ДОБАВЛЕНО: перезапускаем скрипт, чтобы показать кнопку перевода

        # Результат (показывается, если текст уже распознан)
    if st.session_state.english_text:
        st.success("✅ **Распознанный текст:**")
        st.code(st.session_state.english_text, language="text")

        # ДОБАВЛЕНО: кнопка перевода (появляется только после распознавания
        if st.button("🌐 **Перевести на русский**"):
            with st.spinner("Переводим..."):
                st.session_state.russian_text = translate_text(st.session_state.english_text)
            st.rerun()

        # ДОБАВЛЕНО: отображение перевода, если он уже сделан
        if st.session_state.russian_text:
            st.info("🇷🇺 **Перевод на русский:**")
            st.code(st.session_state.russian_text, language="text")

        # Дополнительно показываем предобработанное изображение (для отладки)
        with st.expander("Показать предобработанное изображение (для отладки)"):
            if st.session_state.processed_image:
                st.image(st.session_state.processed_image, caption="Предобработанное изображение", use_container_width=True)

# Информация о модели
st.info("""
**Используемая модель:**  
**microsoft/trocr-base-printed** — одна из лучших открытых моделей для распознавания печатного текста.

TrOCR (Transformer-based OCR) от Microsoft показывает отличные результаты на чётком печатном английском тексте.
""")

# ДОБАВЛЕНО: информационный блок о переводчике
st.info("""
**Используемый переводчик:**  
**Google Translate (через библиотеку deep-translator)** — API для перевода текста.
""")

st.caption("TrOCR + Streamlit • 2026")
