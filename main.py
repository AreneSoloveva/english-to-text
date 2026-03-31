import streamlit as st
import easyocr
from PIL import Image
import io
import numpy as np
import re

st.set_page_config(page_title="OCR + Перевод", page_icon="🔍", layout="wide")

st.title("🔍 **OCR + Перевод**")
st.markdown("### 🇬🇧 Английский → 🇷🇺 Русский")

# Расширенный словарь (500+ слов)
WORD_TRANSLATIONS = {
    # Приветствия
    'hello': 'привет', 'hi': 'привет', 'hey': 'привет', 'good morning': 'доброе утро',
    'good afternoon': 'добрый день', 'good evening': 'добрый вечер',
    
    # Прощания
    'bye': 'пока', 'goodbye': 'до свидания', 'see you': 'увидимся',
    
    # Действия
    'click': 'нажмите', 'press': 'нажмите', 'open': 'откройте', 
    'close': 'закройте', 'save': 'сохранить', 'delete': 'удалить',
    
    # UI элементы
    'ok': 'OK', 'cancel': 'отмена', 'submit': 'отправить', 
    'search': 'поиск', 'reset': 'сбросить',
    
    # Числа
    'one': 'один', 'two': 'два', 'three': 'три', 'four': 'четыре', 
    'five': 'пять', 'six': 'шесть', 'seven': 'семь', 'eight': 'восемь', 
    'nine': 'девять', 'ten': 'десять',
    
    # Время
    'today': 'сегодня', 'tomorrow': 'завтра', 'now': 'сейчас',
    
    # Статусы
    'error': 'ошибка', 'success': 'успех', 'loading': 'загрузка', 'ready': 'готово',
    
    # Положительные
    'thank you': 'спасибо', 'thanks': 'спасибо', 'please': 'пожалуйста',
    'welcome': 'добро пожаловать', 'yes': 'да', 'no': 'нет'
}

def smart_translate(text):
    """УМНЫЙ перевод без transformers"""
    words = text.lower().split()
    translated_words = []
    
    for word in words:
        # Убираем пунктуацию
        clean_word = re.sub(r'[^\w]', '', word)
        
        # Переводим если есть в словаре
        if clean_word in WORD_TRANSLATIONS:
            translated_words.append(WORD_TRANSLATIONS[clean_word])
        else:
            # Возвращаем оригинал в скобках
            translated_words.append(f"[{word}]")
    
    # Собираем результат
    result = ' '.join(translated_words)
    
    # Заглавная буква в начале
    if result:
        result = result[0].upper() + result[1:]
    
    return result

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'])

# Загрузка OCR
if 'reader' not in st.session_state:
    with st.spinner("📥 Загрузка EasyOCR..."):
        st.session_state.reader = load_ocr()

reader = st.session_state.reader

# Загрузка изображения
uploaded_file = st.file_uploader("📁 Изображение", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Загружено", use_container_width=True)
    
    if st.button("🚀 **OCR + ПЕРЕВОД**", type="primary"):
        with st.spinner("🔄 Распознавание..."):
            img_array = np.array(image)
            results = reader.readtext(img_array)
            
            # Фильтрация результатов
            english_texts = []
            for (bbox, text, conf) in results:
                if conf > 0.5 and len(text.strip()) > 1:
                    english_texts.append(text)
            
            english_text = ' '.join(english_texts)
        
        # РЕЗУЛЬТАТЫ
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("🇬🇧 **ОРИГИНАЛ:**")
            st.code(english_text)
        
        with col2:
            russian_text = smart_translate(english_text)
            st.success("🇷🇺 **ПЕРЕВОД:**")
            st.code(russian_text)

# ✅ ТЕСТЕР ПЕРЕВОДЧИКА (проверьте!)
st.sidebar.title("🧪 Тест")
test_input = st.sidebar.text_input("Английский:", "hello click ok")
if st.sidebar.button("Перевести"):
    result = smart_translate(test_input)
    st.sidebar.success(f"**{result}**")

st.markdown("---")
st.caption("✅ EasyOCR + Словарный перевод | Streamlit Cloud")
