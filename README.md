# Распознай английский текст с изображения!

Web-приложение для распознавания текста на английском языке с изображения.
Используются библиотеки:

- [Streamlit](https://streamlit.io/)
- [Transformers](https://huggingface.co/)
- [Torchvision](https://pytorch.org/vision/stable/index.html)
- [deep-translator](https://pypi.org/project/deep-translator/)

Для распознавания изображений используется нейронная сеть [Trocr-base-printed](https://huggingface.co/microsoft/trocr-base-printed). 

## Что добавлено в этом форке: перевод на русский

*   **Функция перевода** распознанного английского текста на русский язык с помощью `deep-translator` (Google Translate API).
*   **Кнопка «Перевести на русский»** в интерфейсе Streamlit — появляется после распознавания и не исчезает при нажатии (использовано `st.session_state`).
*   **Улучшенная обработка длинных текстов** — автоматическое разбиение на части для обхода ограничений переводчика.

[Ссылка на развернутое приложение](https://english-to-text-zespadmyxh6yqwg9nb82ne.streamlit.app/)
