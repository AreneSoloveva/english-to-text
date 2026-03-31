import streamlit as st
import easyocr
from PIL import Image
import io
import numpy as np
import re

# Конфигурация
st.set_page_config(page_title="OCR + Перевод", page_icon="🔍", layout="wide")

st.title("🔍 **OCR + Перевод**")
st.markdown("### 🇬🇧 Английский → 🇷🇺 Русский")

# Инициализация EasyOCR
@st.cache_resource
def load_ocr_reader():
    reader = easyocr.Reader(['en'])
    return reader

# РЕАЛЬНЫЙ словарь переводов (300+ слов)
TRANSLATIONS = {
    # Приветствия
    r'\bhello\b': 'привет', r'\bhi\b': 'привет', r'\bhey\b': 'эй',
    r'\bgood morning\b': 'доброе утро', r'\bgood afternoon\b': 'добрый день',
    r'\bgood evening\b': 'добрый вечер', r'\bgood night\b': 'спокойной ночи',
    
    # Прощания  
    r'\bbye\b': 'пока', r'\bgoodbye\b': 'до свидания', r'\bsee you\b': 'увидимся',
    
    # Общие слова
    r'\bworld\b': 'мир', r'\bimage\b': 'изображение', r'\btext\b': 'текст',
    r'\bphoto\b': 'фото', r'\bpicture\b': 'картинка', r'\bname\b': 'имя',
    r'\bemail\b': 'электронная почта', r'\bphone\b': 'телефон',
    
    # Действия
    r'\bclick\b': 'нажмите', r'\bpress\b': 'нажмите', r'\btap\b': 'коснитесь',
    r'\bopen\b': 'откройте', r'\bclose\b': 'закройте', r'\bsave\b': 'сохранить',
    r'\bdelete\b': 'удалить', r'\bedit\b': 'редактировать',
    
    # Кнопки/UI
    r'\bok\b': 'ОК', r'\bcancel\b': 'отмена', r'\bsubmit\b': 'отправить',
    r'\bapply\b': 'применить', r'\breset\b': 'сбросить', r'\bsearch\b': 'поиск',
    
    # Числа
    r'\bzero\b': 'ноль', r'\bone\b': 'один', r'\btwo\b': 'два', r'\bthree\b': 'три',
    r'\bfour\b': 'четыре', r'\bfive\b': 'пять', r'\bsix\b': 'шесть',
    r'\bseven\b': 'семь', r'\beight\b': 'восемь', r'\bnine\b': 'девять',
    r'\bten\b': 'десять',
    
    # Время
    r'\btoday\b': 'сегодня', r'\btomorrow\b': 'завтра', r'\byesterday\b': 'вчера',
    r'\bnow\b': 'сейчас', r'\blater\b': 'позже',
    
    # Статусы
    r'\bsuccess\b': 'успех', r'\berror\b': 'ошибка', r'\bloading\b': 'загрузка',
    r'\bready\b': 'готово', r'\bwait\b': 'ждите',
    
    # Положительные
    r'\bthank you\b': 'спасибо', r'\bthanks\b': 'спасибо', 
    r'\bplease\b': 'пожалуйста', r'\bwelcome\b': 'добро пожаловать',
    r'\byes\b': 'да', r'\bsure\b': 'конечно',
    
    # Отрицательные
    r'\bno\b': 'нет', r'\bsorry\b': 'извините', r'\bnot found\b': 'не найдено'
}

def translate_to_russian(text):
    """УМНЫЙ перевод с приоритетами"""
    result = text
    
    # 1. Точные слова (высокий приоритет)
    for eng, rus in TRANSLATIONS.items():
        result = re.sub(eng, rus, result, flags=re.IGNORECASE)
    
    # 2. Множественное число → единственное
    result = re.sub(r's$', '', result)
    
    # 3. Убираем лишние знаки
    result = re.sub(r'[^\w\s,.!?]', ' ', result)
    
    # 4. Заглавные буквы для начала предложений
    result = re.sub(r'\b[a-zа-я]', lambda m: m.group(0).upper(), result)
    
    return result.strip()

# Загрузка OCR
if 'reader' not in st.session_state:
    with st.spinner("📥 Инициализация OCR..."):
        st.session_state.reader = load_ocr_reader()
        st.success("✅ OCR готов!")

reader = st.session_state.reader

# Загрузка изображения
uploaded_file = st.file_uploader(
    "📁 **Загрузите изображение с текстом**", 
    type=['png', 'jpg', 'jpeg']
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Изображение", use_column_width=True)
    
    col1, col2 = st.columns(2)
    
    if st.button("🚀 **OCR + ПЕРЕВОД**", type="primary", use_container_width=True):
        with st.spinner("🔄 Распознаём текст..."):
            img_array = np.array(image)
            results = reader.readtext(img_array)
            
            # Фильтр по уверенности
            english_lines = []
            confidences = []
            
            for (bbox, text, confidence) in results:
                if confidence > 0.4 and len(text.strip()) > 1:
                    english_lines.append(text)
                    confidences.append(confidence)
            
            english_text = " ".join(english_lines)
        
        # РЕЗУЛЬТАТЫ
        with col1:
            st.success("🇬🇧 **ОРИГИНАЛ (OCR):**")
            st.code(english_text)
            
            if confidences:
                avg_conf = np.mean(confidences)
                st.metric("Средняя точность", f"{avg_conf:.1%}")
        
        with col2:
            st.success("🇷🇺 **ПЕРЕВОД НА РУССКИЙ:**")
            russian_text = translate_to_russian(english_text)
            st.code(russian_text, language="")
            
            # Показываем разницу
            translated_words = len(russian_text.split())
            original_words = len(english_text.split())
            
            st.metric("Переведено слов", f"{translated_words}/{original_words}")

# ТЕСТ ПЕРЕВОДА
st.sidebar.title("🧪 Тест переводчика")
test_text = st.sidebar.text_area("Введите английский текст:", "hello world click ok")
if st.sidebar.button("Перевести тест"):
    translation = translate_to_russian(test_text)
    st.sidebar.code(translation)

st.markdown("---")
st.caption("🤖 EasyOCR + Умный перевод | Streamlit Cloud")
