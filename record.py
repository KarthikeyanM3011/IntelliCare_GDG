import os
import tempfile
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from PyPDF2 import PdfReader
import google.generativeai as genai
from streamlit_option_menu import option_menu
import requests
from googletrans import Translator
from streamlit_lottie import st_lottie
import time
import string
import random
import requests

languages = [
    'Select Your Language','English','Afrikaans', 'Albanian', 'Amharic', 'Arabic', 'Armenian', 'Azerbaijani', 'Basque', 'Belarusian', 'Bengali', 'Bosnian',
    'Bulgarian', 'Catalan', 'Cebuano', 'Chichewa', 'Chinese (Simplified)', 'Chinese (Traditional)', 'Corsican', 'Croatian',
    'Czech', 'Danish', 'Dutch', 'Esperanto', 'Estonian', 'Filipino', 'Finnish', 'French', 'Frisian', 'Galician',
    'Georgian', 'German', 'Greek', 'Gujarati', 'Haitian Creole', 'Hausa', 'Hawaiian', 'Hebrew', 'Hindi', 'Hmong', 'Hungarian',
    'Icelandic', 'Igbo', 'Indonesian', 'Irish', 'Italian', 'Japanese', 'Javanese', 'Kannada', 'Kazakh', 'Khmer', 'Korean',
    'Kurdish (Kurmanji)', 'Kyrgyz', 'Lao', 'Latin', 'Latvian', 'Lithuanian', 'Luxembourgish', 'Macedonian', 'Malagasy', 'Malay',
    'Malayalam', 'Maltese', 'Maori', 'Marathi', 'Mongolian', 'Myanmar (Burmese)', 'Nepali', 'Norwegian', 'Odia', 'Pashto',
    'Persian', 'Polish', 'Portuguese', 'Punjabi', 'Romanian', 'Russian', 'Samoan', 'Scots Gaelic', 'Serbian', 'Sesotho',
    'Shona', 'Sindhi', 'Sinhala', 'Slovak', 'Slovenian', 'Somali', 'Spanish', 'Sundanese', 'Swahili', 'Swedish', 'Tajik',
    'Tamil', 'Telugu', 'Thai', 'Turkish', 'Ukrainian', 'Urdu', 'Uyghur', 'Uzbek', 'Vietnamese', 'Welsh', 'Xhosa', 'Yiddish', 'Yoruba','Zulu'
]

LANGUAGES = {
'Afrikaans': 'af',
'Albanian': 'sq',
'Amharic': 'am',
'Arabic': 'ar',
'Armenian': 'hy',
'Azerbaijani': 'az',
'Basque': 'eu',
'Belarusian': 'be',
'Bengali': 'bn',
'Bosnian': 'bs',
'Bulgarian': 'bg',
'Catalan': 'ca',
'Cebuano': 'ceb',
'Chichewa': 'ny',
'Chinese (Simplified)': 'zh-cn',
'Chinese (Traditional)': 'zh-tw',
'Corsican': 'co',
'Croatian': 'hr',
'Czech': 'cs',
'Danish': 'da',
'Dutch': 'nl',
'English': 'en',
'Esperanto': 'eo',
'Estonian': 'et',
'Filipino': 'tl',
'Finnish': 'fi',
'French': 'fr',
'Frisian': 'fy',
'Galician': 'gl',
'Georgian': 'ka',
'German': 'de',
'Greek': 'el',
'Gujarati': 'gu',
'Haitian Creole': 'ht',
'Hausa': 'ha',
'Hawaiian': 'haw',
'Hebrew': 'iw',
'Hindi': 'hi',
'Hmong': 'hmn',
'Hungarian': 'hu',
'Icelandic': 'is',
'Igbo': 'ig',
'Indonesian': 'id',
'Irish': 'ga',
'Italian': 'it',
'Japanese': 'ja',
'Javanese': 'jw',
'Kannada': 'kn',
'Kazakh': 'kk',
'Khmer': 'km',
'Korean': 'ko',
'Kurdish (Kurmanji)': 'ku',
'Kyrgyz': 'ky',
'Lao': 'lo',
'Latin': 'la',
'Latvian': 'lv',
'Lithuanian': 'lt',
'Luxembourgish': 'lb',
'Macedonian': 'mk',
'Malagasy': 'mg',
'Malay': 'ms',
'Malayalam': 'ml',
'Maltese': 'mt',
'Maori': 'mi',
'Marathi': 'mr',
'Mongolian': 'mn',
'Myanmar (Burmese)': 'my',
'Nepali': 'ne',
'Norwegian': 'no',
'Odia': 'or',
'Pashto': 'ps',
'Persian': 'fa',
'Polish': 'pl',
'Portuguese': 'pt',
'Punjabi': 'pa',
'Romanian': 'ro',
'Russian': 'ru',
'Samoan': 'sm',
'Scots Gaelic': 'gd',
'Serbian': 'sr',
'Sesotho': 'st',
'Shona': 'sn',
'Sindhi': 'sd',
'Sinhala': 'si',
'Slovak': 'sk',
'Slovenian': 'sl',
'Somali': 'so',
'Spanish': 'es',
'Sundanese': 'su',
'Swahili': 'sw',
'Swedish': 'sv',
'Tajik': 'tg',
'Tamil': 'ta',
'Telugu': 'te',
'Thai': 'th',
'Turkish': 'tr',
'Ukrainian': 'uk',
'Urdu': 'ur',
'Uyghur': 'ug',
'Uzbek': 'uz',
'Vietnamese': 'vi',
'Welsh': 'cy',
'Xhosa': 'xh',
'Yiddish': 'yi',
'Yoruba': 'yo',
'Zulu': 'zu'
}

GOOGLE_API_KEY = 'AIzaSyAPMmJVLK403FPhDjNCz2f6cbZWVWROGLg'
genai.configure(api_key=GOOGLE_API_KEY)

model_name = 'gemini-1.0-pro'
client = genai.GenerativeModel(model_name)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

def generate_random_string(length=10):
    characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string

def rag_implement_retrive_request(user_id, collection_id, question):
    url = "http://127.0.0.1:5000/rag_implement_retrive"
    payload = {
        "user_id": user_id,
        "collection_id": collection_id,
        "question": question
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise an error if the request failed
        print("Retrieve Answer Response:", response.json())
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error retrieving answer: {e}")
        return None
    
def process_pdf_request(text, user_id, collection_id):
    url = "http://127.0.0.1:5000/process_pdf"
    payload = {
        "text": text,
        "user_id": user_id,
        "collection_id": collection_id
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise an error if the request failed
        print("Process PDF Response:", response.json())
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error processing PDF: {e}")
        return None

def generate_chat_response(message):
    try:
        response = rag_implement_retrive_request(st.session_state.user_id, st.session_state.collection_id, message)
        if response is None:
            return "No related content is found. Ask questions only frorm the uploaded File"
        else:
            result = response['response']
        return result
    except Exception as e:
        return "Try again after sometimes"
    
def chat_interface():
    st.subheader("Chat with PDF Report")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Type your message:", key="user_input")

    if st.button("Send"):
        if user_input:
            st.session_state.chat_history.append(("🙋", user_input))

            response = generate_chat_response(user_input)
            st.session_state.chat_history.append(("🤖", response))

            # st.rerun()

    for sender, message in st.session_state.chat_history:
        if sender == "You":
            st.write(f"**{sender}:** {message}")
        else:
            st.write(f"**{sender}:** {message}", key=f"{time.time()}")

def extract_pdf(path):
    reader = PdfReader(path)
    extracted_text = ""
    for page in reader.pages:
        extracted_text += page.extract_text() or ""
    return extracted_text


def scanner():
    if "user_id" not in st.session_state:
        st.session_state.user_id = generate_random_string()

    if "collection_ids" not in st.session_state:
        st.session_state.collection_ids = []

    def load_lottieurl(url: str):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    def extract_pdf(path):
        reader = PdfReader(path)
        extracted_text = ""
        for page in reader.pages:
            extracted_text += page.extract_text() or ""
        return extracted_text
    
    def translate(info,lang):
            translator=Translator()
            translation = translator.translate(info, dest=LANGUAGES[lang])
            return translation.text

    left_column, right_column = st.columns((2, 5))

    with left_column:
        logo = 'https://lottie.host/49cfa049-139a-498a-954f-7985b2b60086/qvfWaOHQJR.json'
        logo_image = load_lottieurl(logo)
        st_lottie(logo_image, width=300, height=100, key='logo')

    with right_column:
        st.header("RecordChat")
    selected_language = st.selectbox("Select your preferred language:", languages)
    if selected_language!='Select Your Language':
        selected_language_code_rec = selected_language
        pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
        report_res=''
        if pdf_file is not None:
            try:
                query = extract_pdf(pdf_file)
                if "pdf_report" not in st.session_state or st.session_state.pdf_report["content"]!=query:
                    if "collection_id" not in st.session_state:
                        random_id = generate_random_string()
                        st.session_state.collection_id = random_id
                        st.session_state.collection_ids.append(random_id)
                    prefix = 'You are a helpful medical assistant. From the Patient details given below for your reference, you need to generate a detailed report about it and explain the terms in that and other stuff mentioned in the medical report content. You must give only the detailed explanation of the given content in a complete report structure.Give the response in a prettified manner. **Medical report content**: '
                    print("generating")
                    report_res = client.generate_content(prefix + query)
                    process_pdf_request(query, st.session_state.user_id, st.session_state.collection_id)
                    print("generated")
                    report_res = report_res.text
                    st.session_state.pdf_report = {"content":query, "report":report_res}
                    st.session_state.chat_history = []
                else:
                    report_res = st.session_state.pdf_report["report"]
            except Exception as e:
                report_res = 'Error Occured while generating'
                print(e)
            finally:
                with st.expander(translate("Report Overview", selected_language_code_rec)):
                    answer_placeholder=st.empty()
                    answer_placeholder.write(translate(report_res, selected_language_code_rec))
                chat_interface()

def report():
    scanner()