import io
import streamlit as st
import torch
from transformers import AutoProcessor, GlmOcrForConditionalGeneration
from PIL import Image
from transformers import pipeline

# Функция загрузки модели OCR
@st.cache_resource
def load_ocr_model():
    model_id = "zai-org/GLM-OCR"
    processor = AutoProcessor.from_pretrained(model_id)
    model = GlmOcrForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    return processor, model

# Функция загрузки переводчика
@st.cache_resource
def load_translator():
    translator = pipeline(
        "translation",
        model="Helsinki-NLP/opus-mt-en-ru",
        device_map="auto" if torch.cuda.is_available() else -1
    )
    return translator

def load_image():
    uploaded_file = st.file_uploader('Выберите изображение для распознавания', type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data)).convert('RGB')
    return None

# Заголовок приложения
st.title('🌐 OCR + Перевод текста с изображений (GLM-OCR)')

# Загрузка моделей
ocr_processor, ocr_model = load_ocr_model()
translator = load_translator()

# Загрузка изображения
img = load_image()

# Кнопка распознавания
if st.button('🔍 Распознать и перевести') and img is not None:
    with st.spinner('Распознавание текста...'):
        # Подготовка сообщения для GLM-OCR
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": "Recognize all the text in this image."},
                ],
            }
        ]
        
        # Применение chat template
        inputs = ocr_processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True
        )
        
        # Генерация текста
        with torch.no_grad():
            generated_ids = ocr_model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=0.1
            )
        
        generated_text = ocr_processor.decode(generated_ids[0], skip_special_tokens=True)
        
        st.success('✅ Текст распознан!')
        st.markdown('**📄 Распознанный английский текст:**')
        st.markdown(f'`{generated_text}`')
        
        # Перевод на русский
        with st.spinner('Перевод на русский...'):
            translation = translator(generated_text)[0]['translation_text']
        
        st.markdown('**🇷🇺 Перевод на русский:**')
        st.markdown(f'`{translation}`')

# Инфо о модели
with st.expander("ℹ️ Информация о моделях"):
    st.info("""
    **OCR модель:** zai-org/GLM-OCR  
    Поддерживает английский, китайский, русский, японский и другие языки.  
    **Переводчик:** Helsinki-NLP/opus-mt-en-ru  
    Точный перевод с английского на русский.
    """)
