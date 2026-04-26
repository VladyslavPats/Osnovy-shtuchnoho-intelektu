import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if api_key:
    genai.configure(api_key=api_key)
    try:
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        active_model_name = available_models[0] if available_models else "gemini-1.5-flash"
    except Exception:
        active_model_name = "gemini-1.5-flash"
else:
    st.error("GEMINI_API_KEY не знайдено у файлі .env")
    st.stop()

st.set_page_config(page_title="AI English Tutor", layout="wide")


def extract_text_from_pdfs(files):
    combined_text = ""
    for pdf_file in files:
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            page_content = page.extract_text()
            if page_content:
                combined_text += page_content
    return combined_text


def create_text_chunks(text_data):
    splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    return splitter.split_text(text_data)


def create_vector_store(chunks):
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    store = FAISS.from_texts(chunks, embedding=embedding_model)
    return store


with st.sidebar:
    st.header("Навчальні матеріали")
    uploaded_pdfs = st.file_uploader("Завантажте PDF-підручники", accept_multiple_files=True, type=["pdf"])

    if st.button("Обробити базу знань"):
        if uploaded_pdfs:
            with st.spinner("Аналіз лінгвістичних даних..."):
                raw_text = extract_text_from_pdfs(uploaded_pdfs)
                chunks = create_text_chunks(raw_text)
                v_store = create_vector_store(chunks)
                st.session_state.vector_store = v_store
                st.success("База знань готова")
        else:
            st.warning("Завантажте PDF-файл")

    st.divider()
    if st.button("Очистити історію"):
        st.session_state.messages = []
        if "vector_store" in st.session_state:
            del st.session_state.vector_store
        st.rerun()

st.title("AI English Learning Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Напишіть ваше питання..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        if "vector_store" in st.session_state:
            try:
                related_docs = st.session_state.vector_store.similarity_search(user_input)
                context_data = "\n".join([doc.page_content for doc in related_docs])

                instruction = """
                ТИ СПЕЦІАЛІЗОВАНИЙ ШІ-РЕПЕТИТОР З АНГЛІЙСЬКОЇ МОВИ.
                ПРАВИЛА:
                1. Відповідай ТІЛЬКИ на питання, що стосуються вивчення англійської мови.
                2. Використовуй ТІЛЬКИ наданий контекст (текст підручника).
                3. Якщо питання не стосується англійської мови, ввічливо відмовся відповідати, пояснивши, що ти працюєш лише з підручником англійської.
                4. Якщо в підручнику немає відповіді на питання по англійській мові, так і скажи.
                """
                prompt_content = f"{instruction}\n\nКОНТЕКСТ З ПІДРУЧНИКА:\n{context_data}\n\nПИТАННЯ СТУДЕНТА: {user_input}"

                ai_model = genai.GenerativeModel(active_model_name)
                ai_response = ai_model.generate_content(prompt_content)
                final_text = ai_response.text

            except Exception as err:
                final_text = f"Виникла технічна помилка: {err}"
        else:
            final_text = "Будь ласка, спочатку завантажте підручник з англійської мови у форматі PDF та натисніть кнопку 'Обробити базу знань'."

        st.markdown(final_text)
        st.session_state.messages.append({"role": "assistant", "content": final_text})