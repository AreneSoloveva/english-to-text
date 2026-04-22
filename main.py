import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import re
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
import numpy as np
import easyocr
from deep_translator import GoogleTranslator

# Кэшируем модель (загружается один раз при запуске)
@st.cache_resource
def load_trocr():
    trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
    return trocr_processor, trocr_model

# ДОБАВЛЕНО: кэширование модели с детектером
@st.cache_resource
def load_easyocr():
    # EasyOCR с детектором CRAFT (по умолчанию)
    return easyocr.Reader(['en'], gpu=False)

# ДОБАВЛЕНО: кэширование переводчика
@st.cache_resource
def get_translator():
    return GoogleTranslator(source='en', target='ru')

# Загружаем модель
trocr_processor, trocr_model = load_trocr()
easyocr_reader = load_easyocr()
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

# ДОБАВЛЕНО: группировка боксов строки
def group_boxes_by_line(boxes_with_centers, y_threshold=15):
    """
    boxes_with_centers: list of (bbox, center_x, center_y)
    Возвращает list of list of bbox (каждый внутренний список — одна строка текста).
    """
    if not boxes_with_centers:
        return []
    # Сортируем по Y (центру)
    boxes_with_centers.sort(key=lambda x: x[2])
    lines = []
    current_line = [boxes_with_centers[0]]
    for box in boxes_with_centers[1:]:
        # Если разница по Y с первым боксом текущей строки меньше порога
        if abs(box[2] - current_line[0][2]) <= y_threshold:
            current_line.append(box)
        else:
            # Сортируем текущую строку по X (слева направо)
            current_line.sort(key=lambda x: x[1])
            lines.append([b[0] for b in current_line])  # сохраняем только bbox
            current_line = [box]
    if current_line:
        current_line.sort(key=lambda x: x[1])
        lines.append([b[0] for b in current_line])
    return lines

# ДОБАВЛЕНО: детекция (CRAFT) + распознавание (TrOCR)
def detect_and_recognize(image):
    """
    Принимает PIL Image.
    Возвращает распознанный английский текст (строка).
    """
    # Конвертируем PIL -> numpy (EasyOCR)
    img_np = np.array(image.convert('RGB'))
    # Получаем боксы (detail=1 возвращает (bbox, text, confidence))
    results = easyocr_reader.readtext(img_np, paragraph=False, detail=1)
    if not results:
        return ""

    # Вычисляем центры боксов
    boxes_with_centers = []
    for (bbox, _, _) in results:
        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]
        center_x = sum(x_coords) / 4
        center_y = sum(y_coords) / 4
        boxes_with_centers.append((bbox, center_x, center_y))

    # Группируем в строки
    lines = group_boxes_by_line(boxes_with_centers, y_threshold=15)

    recognized_lines = []
    for line_boxes in lines:
        # Объединяем все боксы строки в одну область
        all_x = []
        all_y = []
        for bbox in line_boxes:
            for (x, y) in bbox:
                all_x.append(x)
                all_y.append(y)
        x_min = int(min(all_x))
        x_max = int(max(all_x))
        y_min = int(min(all_y))
        y_max = int(max(all_y))

        # Вырезаем область строки
        cropped = image.crop((x_min, y_min, x_max, y_max))

        # Распознаём через TrOCR
        pixel_values = trocr_processor(images=cropped, return_tensors="pt").pixel_values
        with torch.no_grad():
            generated_ids = trocr_model.generate(pixel_values, max_new_tokens=128)
        line_text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        recognized_lines.append(line_text.strip())

    # Склеиваем строки через пробел
    full_text = ' '.join(recognized_lines)
    full_text = re.sub(r'\s+', ' ', full_text).strip()
    return full_text

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
st.markdown("### Распознавание печатного английского текста\n**Детектор:** CRAFT (через EasyOCR) | **Распознаватель:** `microsoft/trocr-base-printed`")


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
        with st.spinner("🔄 Детекция (CRAFT) + распознавание (TrOCR)..."):
            # Предобработка
            processed = preprocess_image(image)
            english_text = detect_and_recognize(processed)

        if not english_text:
            st.warning("⚠️ Текст не найден на изображении.")
            st.session_state.english_text = None
        else:
            st.session_state.english_text = english_text
            st.session_state.processed_image = processed
            st.session_state.russian_text = None
        st.rerun()

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
**Используемы модели:**  
**microsoft/trocr-base-printed** — модель для распознавания печатного текста.

TrOCR (Transformer-based OCR) от Microsoft показывает отличные результаты на чётком печатном английском тексте.

**CRAFT** (через EasyOCR) — детектор текста, находит строки.
""")

# ДОБАВЛЕНО: информационный блок о переводчике
st.info("""
**Используемый переводчик:**  
**Google Translate (через библиотеку deep-translator)** — API для перевода текста.
""")

st.caption("TrOCR + Streamlit • 2026")
