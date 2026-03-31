import streamlit as st
import easyocr
from PIL import Image
import io
import numpy as np
import re

# Конфигурация
st.set_page_config(
    page_title="OCR + Перевод", 
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 **OCR + Перевод**")
st.markdown("### 🇬🇧 Английский → 🇷🇺 Русский | Работает на Streamlit Cloud!")

# Инициализация EasyOCR (кэшируется)
@st.cache_resource
def load_ocr_reader():
    """Загрузка EasyOCR для английского"""
    reader = easyocr.Reader(['en'])
    return reader

# Простой переводчик (словарный)
def translate_to_russian(text):
    """Базовый перевод en-ru без torch"""
    translations = {
        # Общие слова
        r'\bhello\b': 'привет',
        r'\bhi\b': 'привет', 
        r'\bworld\b': 'мир',
        r'image\b': 'изображение',
        r'text\b': 'текст',
        r'welcome\b': 'добро пожаловать',
        r'thank\b': 'спасибо',
        r'please\b': 'пожалуйста',
        r'good\b': 'хорошо',
        r'yes\b': 'да',
        r'no\b': 'нет',
        r'name\b': 'имя',
        r'email\b': 'почта',
        
        # Числа
        r'\bone\b': 'один',
        r'\btwo\b': 'два',
        r'\bthree\b': 'три',
        r'\bfour\b': 'четыре',
        r'\bfive\b': 'пять',
        
        # Технические
        r'click\b': 'нажмите',
        r'button\b': 'кнопка',
        r'open\b': 'открыть',
        r'close\b': 'закрыть',
        r'save\b': 'сохранить'
    }
    
    result = text
    for eng, rus in translations.items():
        result = re.sub(eng, rus, result, flags=re.IGNORECASE)
    
    return result

# Загрузка OCR
progress = st.progress(0)
status = st.empty()

status.text("📥 Инициализация OCR...")
reader = load_ocr_reader()
progress.progress(100)
status.success("✅ OCR готов!")

# UI
col1, col2 = st.columns([1, 2])

with col1:
    st.info("**🚀 EasyOCR + словарный перевод**")
    st.info("**Поддержка:** английский текст")

uploaded_file = st.file_uploader(
    "📁 Загрузите изображение", 
    type=['png', 'jpg', 'jpeg']
)

if uploaded_file is not None:
    # Показ изображения
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Изображение", use_column_width=True)
    
    # Кнопка OCR
    if st.button("🚀 **РАСПОЗНАТЬ + ПЕРЕВЕСТИ**", type="primary"):
        with st.spinner("🔄 Распознавание текста..."):
            # Конвертация в numpy
            img_array = np.array(image)
            
            # EasyOCR распознавание
            results = reader.readtext(img_array)
            
            # Извлечение текста
            english_lines = []
            for (bbox, text, confidence) in results:
                if confidence > 0.5:  # Только уверенные результаты
                    english_lines.append(text)
            
            english_text = " ".join(english_lines)
        
        # Результаты в колонках
        col_eng, col_ru = st.columns(2)
        
        with col_eng:
            st.success("🇬🇧 **English OCR:**")
            st.code(english_text)
            st.caption(f"Найдено строк: {len(english_lines)}")
        
        with col_ru:
            russian_text = translate_to_russian(english_text)
            st.success("🇷🇺 **Русский перевод:**")
            st.code(russian_text)
        
        # Статистика
        st.metric("Точность", f"{np.mean([conf for _,_,conf in results]):.1%}")

# Инструкции
with st.expander("📖 Как использовать"):
    st.markdown("""
    1. Загрузите **изображение с английским текстом**
    2. Нажмите **"РАСПОЗНАТЬ + ПЕРЕВЕСТИ"**
    3. Получите **OCR + перевод**!
    
    ✅ **Работает:** четкий печатный текст
    ⚠️ **Лучше:** крупный шрифт, хороший контраст
    """)

st.markdown("---")
st.caption("🤖 EasyOCR | Streamlit Cloud | CPU-only")
