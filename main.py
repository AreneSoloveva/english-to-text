import streamlit as st
from PIL import Image
import io
import numpy as np

st.set_page_config(
    page_title="OCR Demo", 
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 **Image OCR Demo**")
st.markdown("### Загрузите изображение с текстом")

# Сайдбар
st.sidebar.info("✅ **Статус:** Готов к работе")
st.sidebar.caption("Streamlit Cloud Edition")

# Загрузка файла
uploaded_file = st.file_uploader(
    "📁 Выберите изображение", 
    type=['png', 'jpg', 'jpeg']
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Загружено", use_column_width=True)
    
    # Информация
    st.success("✅ Изображение загружено!")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Размер", f"{image.size[0]}x{image.size[1]}")
        st.metric("Формат", image.format or "Неизвестно")
    
    with col2:
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        st.metric("Размер файла", f"{len(buffer.getvalue())//1024} KB")
    
    # Кнопка "OCR" (заглушка)
    if st.button("🚀 **Запустить OCR**", type="primary"):
        st.info("🔄 **OCR недоступен в бесплатной версии Streamlit Cloud**")
        st.markdown("""
        **Почему нет OCR:**
        - ❌ torch не устанавливается (Python 3.14.3 + 403 Forbidden)
        - ❌ Нет GPU/CPU оптимизации
        - ✅ **Решение:** Google Colab или локальный запуск
        """)

# Footer
st.markdown("---")
st.markdown("*🤖 Demo версия | Полный OCR → Colab*")
