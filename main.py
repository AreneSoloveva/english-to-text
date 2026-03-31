import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
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
    st.image(image, caption="Оригинал", use_container_width=True)
    
    if st.button("🚀 **Распознать текст**", type="primary"):
        with st.spinner("🔄 TrOCR распознаёт текст... (это может занять 5–15 сек)"):
            # Предобработка
            processed = preprocess_image(image)
            
            # Распознавание с помощью TrOCR
            pixel_values = processor(images=processed, return_tensors="pt").pixel_values
            generated_ids = model.generate(pixel_values, max_new_tokens=100)
            english_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Очистка текста
            english_text = re.sub(r'\s+', ' ', english_text).strip()
        
        # Вывод результата
        st.success("🇬🇧 **Распознанный английский текст:**")
        st.code(english_text, language="text")

st.info("""
**Используется модель:** microsoft/trocr-base-printed  
✅ Хорошо работает с печатным английским текстом  
⚠️ На Streamlit Cloud используется CPU-версия (работает медленнее)
""")

st.caption("TrOCR OCR | Streamlit Cloud 2026")
