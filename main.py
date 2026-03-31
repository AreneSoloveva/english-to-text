import streamlit as st
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import io
import numpy as np

# Конфигурация
st.set_page_config(
    page_title="TrOCR OCR", 
    page_icon="🔍", 
    layout="wide"
)

st.title("🔍 **TrOCR OCR**")
st.markdown("**microsoft/trocr-base-printed** [Hugging Face](https://huggingface.co/microsoft/trocr-base-printed)")

# Загрузка модели TrOCR
@st.cache_resource
def load_trocr_model():
    """Загрузка microsoft/trocr-base-printed"""
    st.info("📥 Загружаем TrOCR-base-printed (~300MB)...")
    
    processor = TrOCRProcessor.from_pretrained(
        "microsoft/trocr-base-printed",
        trust_remote_code=False
    )
    
    model = VisionEncoderDecoderModel.from_pretrained(
        "microsoft/trocr-base-printed",
        torch_dtype=torch.float32,  # CPU-only
        low_cpu_mem_usage=True
    )
    
    return processor, model

# Прогресс-бар загрузки
progress_bar = st.progress(0)
status_text = st.empty()

status_text.text("🔄 Инициализация TrOCR...")
processor, model = load_trocr_model()
progress_bar.progress(100)
status_text.success("✅ **TrOCR-base-printed готов!**")

# Сайдбар с инфо
with st.sidebar:
    st.markdown("### ℹ️ **Модель:**")
    st.info("**microsoft/trocr-base-printed**")
    st.markdown("- 🎯 Печатный текст")
    st.markdown("- 💾 ~300MB") 
    st.markdown("- 🖥️ CPU-оптимизирована")
    st.caption("https://huggingface.co/microsoft/trocr-base-printed")

# Загрузка изображения
uploaded_file = st.file_uploader(
    "📁 **Загрузите изображение с текстом**", 
    type=['png', 'jpg', 'jpeg', 'webp'],
    help="Лучше всего работает с печатным текстом"
)

if uploaded_file is not None:
    # Показ изображения
    image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
    st.image(image, caption="Загруженное изображение", use_column_width=True)
    
    # Кнопка распознавания
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🚀 **РАСПОЗНАТЬ ТЕКСТ**", type="primary", use_container_width=True):
            with st.spinner("🔄 TrOCR распознаёт текст..."):
                # Обработка изображения
                pixel_values = processor(image, return_tensors="pt").pixel_values
                
                # Генерация текста (CPU)
                with torch.no_grad():
                    generated_ids = model.generate(
                        pixel_values,
                        max_length=128,
                        num_beams=4,
                        do_sample=False,
                        early_stopping=True,
                        pad_token_id=processor.tokenizer.eos_token_id
                    )
                
                # Декодирование результата
                extracted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Результат
            with col2:
                st.success("✅ **Распознанный текст:**")
                st.code(extracted_text, language="")
                
                # Статистика
                word_count = len(extracted_text.split())
                st.metric("Слов", word_count)
                st.metric("Символов", len(extracted_text))

# Инструкции
with st.expander("📖 Инструкция"):
    st.markdown("""
    ### 🎯 **Лучшие результаты:**
    - ✅ **Печатный текст** (книги, документы)
    - ✅ **Крупный шрифт**
    - ✅ **Хороший контраст**
    
    ### ⚠️ **Слабо работает:**
    - ❌ Рукописный текст
    - ❌ Маленький шрифт  
    - ❌ Плохое освещение
    """)

st.markdown("---")
st.caption("🤖 **TrOCR-base-printed** | Streamlit Cloud 2026")
