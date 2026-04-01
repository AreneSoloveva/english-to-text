import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import re
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

# Кэшируем модель (загружается один раз при запуске)
@st.cache_resource
def load_trocr_model():
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
    return processor, model


# Загружаем модель
processor, model = load_trocr_model()

st.set_page_config(
    page_title="TrOCR OCR",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 **TrOCR OCR**")
st.markdown("### Распознавание печатного английского текста\n**Модель:** `microsoft/trocr-base-printed`")

# Предобработка изображения (оптимально для TrOCR)
def preprocess_image(image):
    """Увеличение контраста + медианный фильтр"""
    enhancer = ImageEnhance.Contrast(image)
    enhanced = enhancer.enhance(2.0)
    enhanced = enhanced.filter(ImageFilter.MedianFilter())
    return enhanced


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

        # Результат
        st.success("✅ **Распознанный текст:**")
        st.code(english_text, language="text")

        # Дополнительно показываем предобработанное изображение (для отладки)
        with st.expander("Показать предобработанное изображение (для отладки)"):
            st.image(processed_image, caption="Предобработанное изображение", use_container_width=True)

# Информация о модели
st.info("""
**Используемая модель:**  
**microsoft/trocr-base-printed** — одна из лучших открытых моделей для распознавания печатного текста (printed text).

TrOCR (Transformer-based OCR) от Microsoft показывает отличные результаты на чётком печатном английском тексте.
""")

st.caption("TrOCR + Streamlit • 2026")
