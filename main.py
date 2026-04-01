import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import re
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
from difflib import SequenceMatcher

# ====================== ДОСТУПНЫЕ МОДЕЛИ ======================
AVAILABLE_MODELS = {
    "Английский (оригинал)": "microsoft/trocr-base-printed",
    "Немецкий (German)": "microsoft/trocr-base-printed",           # пока используем базовую, можно заменить на fine-tuned позже
    "Французский (French)": "agomberto/trocr-base-printed-fr",     # популярная fine-tuned версия
    "Испанский (Spanish)": "qantev/trocr-base-spanish"             # хорошая fine-tuned версия для испанского
}

# Кэшируем загрузку моделей
@st.cache_resource
def load_model(model_name: str):
    st.info(f"Загружаем модель: **{model_name}** ...", icon="⏳")
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    return processor, model


st.set_page_config(
    page_title="TrOCR OCR — Multilingual",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 **TrOCR OCR** — Распознавание текста")
st.markdown("### Поддержка английского, немецкого, французского и испанского языков")

# ====================== ВЫБОР МОДЕЛИ ======================
st.sidebar.header("⚙️ Настройки модели")
selected_language = st.sidebar.selectbox(
    "Выберите язык:",
    options=list(AVAILABLE_MODELS.keys()),
    index=0
)

model_name = AVAILABLE_MODELS[selected_language]

# Загружаем выбранную модель
processor, model = load_model(model_name)

# ====================== ПРЕДОБРАБОТКА ======================
def preprocess_image(image):
    enhancer = ImageEnhance.Contrast(image)
    enhanced = enhancer.enhance(2.0)
    enhanced = enhanced.filter(ImageFilter.MedianFilter())
    return enhanced


# ====================== ОСНОВНОЙ ИНТЕРФЕЙС ======================
uploaded_file = st.file_uploader(
    "📁 Загрузите изображение с текстом",
    type=['png', 'jpg', 'jpeg', 'tiff']
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📸 Оригинальное изображение", use_container_width=True)

    expected_text = st.text_area(
        "Ожидаемый текст (для сравнения):",
        height=80,
        placeholder="Вставьте сюда правильный текст с изображения..."
    )

    # Кнопка распознавания
    if st.button("🚀 **Распознать текст**", type="primary", use_container_width=True):
        with st.spinner(f"🔄 Распознавание с помощью модели **{selected_language}**..."):
            processed = preprocess_image(image)

            pixel_values = processor(images=processed, return_tensors="pt").pixel_values

            with torch.no_grad():
                generated_ids = model.generate(
                    pixel_values,
                    max_new_tokens=512,
                    num_beams=5,
                    early_stopping=True,
                    length_penalty=1.0
                )

            recognized_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            recognized_text = re.sub(r'\s+', ' ', recognized_text).strip()

        # Результат
        st.success(f"**Распознанный текст ({selected_language})**")
        st.code(recognized_text, language="text")

        # Сравнение, если пользователь ввёл ожидаемый текст
        if expected_text.strip():
            similarity = SequenceMatcher(None, recognized_text.lower(), expected_text.lower()).ratio()
            cer_approx = 1 - similarity

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Схожесть", f"{similarity:.1%}")
            with col2:
                st.metric("Примерный CER", f"{cer_approx:.1%}")

            st.info(f"**Ожидаемый текст:**\n{expected_text}")

        with st.expander("Предобработанное изображение (отладка)"):
            st.image(processed, use_container_width=True)

# ====================== ИНФОРМАЦИЯ ======================
st.info(f"""
**Текущая модель:** `{model_name}`

- **Английский** — отличное качество (оригинальная модель)
- **Немецкий** — среднее качество
- **Французский** — улучшено благодаря fine-tuned модели
- **Испанский** — улучшено благодаря fine-tuned модели

Для лучших результатов на немецком можно позже добавить специализированные fine-tuned модели.
""")

st.caption("TrOCR • Streamlit • Поддержка нескольких языков • 2026")
st.caption("TrOCR + Streamlit • 2026")
