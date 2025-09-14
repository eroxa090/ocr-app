import streamlit as st
from paddleocr import PaddleOCR
import re
import os
from datetime import datetime
from openai import OpenAI

# ------------------------------
# Настройка OCR
# ------------------------------
ocr = PaddleOCR(lang='ru')  # OCR с поддержкой русского языка

# ------------------------------
# Настройка OpenAI
# ------------------------------
# ⚠️ Перед запуском в консоли:
# export OPENAI_API_KEY="твой_ключ"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ------------------------------
# Функции
# ------------------------------
def extract_text(image_path):
    """Извлекаем текст через PaddleOCR"""
    result = ocr.ocr(image_path, cls=True)
    text = "\n".join([line[1][0] for line in result[0]])
    return text

def llm_postprocess(text):
    """Обработка текста через GPT (получаем JSON)"""
    prompt = f"""
    Вот текст банковского документа (распознанный OCR):
    {text}

    Извлеки следующие поля:
    - ФИО
    - Дата (в формате дд.мм.гггг)
    - Сумма
    - Валюта

    Верни результат строго в JSON.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",   # можно заменить на gpt-3.5-turbo
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

def simple_extract_fields(text):
    """Запасной вариант (регулярки вместо GPT)"""
    data = {}

    # Дата
    date_match = re.search(r"(\d{2}[./-]\d{2}[./-]\d{2,4})", text)
    if date_match:
        data["Дата"] = date_match.group(1)

    # Сумма
    sum_match = re.search(r"([\d\s]+[,.]?\d*)\s?(тг|₸|KZT|руб|₽)?", text, re.IGNORECASE)
    if sum_match:
        data["Сумма"] = sum_match.group(1).replace(" ", "")
        data["Валюта"] = sum_match.group(2) if sum_match.group(2) else "₸"

    # ФИО
    fio_match = re.search(r"(?:ФИО|Имя)\s*[:\-]?\s*([А-ЯЁ][а-яё]+\s[А-ЯЁ]\.[А-ЯЁ]\.)", text)
    if fio_match:
        data["ФИО"] = fio_match.group(1)

    return data

# ------------------------------
# Интерфейс Streamlit
# ------------------------------
st.title("📄 OCR 2.0 для банковских документов")

uploaded_file = st.file_uploader("Загрузите чек или документ", type=["jpg", "png", "jpeg"])

use_llm = st.checkbox("🤖 Использовать Ai для пост-обработки", value=True)

if uploaded_file is not None:
    # Сохраняем загруженный файл
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Шаг 1: OCR
    raw_text = extract_text("temp.jpg")
    st.subheader("📜 Распознанный текст (OCR):")
    st.write(raw_text)

    # Шаг 2: Извлечение JSON
    if use_llm:
        st.subheader("📂 JSON (GPT):")
        json_result = llm_postprocess(raw_text)
        st.code(json_result, language="json")
    else:
        st.subheader("📂 JSON (правила):")
        fields = simple_extract_fields(raw_text)
        st.json(fields)
