import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import streamlit as st
from streamlit_option_menu import option_menu
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner
from googletrans import Translator
# from sentence_transformers import SentenceTransformer
from langchain.prompts import PromptTemplate
import google.generativeai as genai
import PyPDF2
from pinecone import Pinecone, ServerlessSpec
import requests
import keras_ocr
import pandas as pd
import time
import numpy as np
import cv2
from io import BytesIO
import base64
from googletrans import Translator
from record import *
from huggingface_hub import InferenceClient

# def generate_response_chatbot(prompt):
#     try:
#         client = InferenceClient(api_key="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")

#         messages = [
#             {
#                 "role": "user",
#                 "content": prompt
#             }
#         ]

#         stream = client.chat.completions.create(
#             model="mistralai/Mixtral-8x7B-Instruct-v0.1", 
#             messages=messages, 
#             max_tokens=500,
#             stream=True
#         )

#         res = ""
#         for chunk in stream:
#             res += chunk.choices[0].delta.content  
#         return res
#     except Exception as e:
#         print(e)
#         return "Something went wrong! Try again after sometime!"

# def format_query_response_chatbot(user_query):
#     prompt = f"""
#     You are a highly trained medical assistant bot. It is extremely important that you provide **only accurate, well-founded, and medically reliable responses** to the user's queries.
#     You should not make assumptions or provide vague, incomplete, or speculative answers. Your response must be based on well-established medical knowledge and principles.
    
#     If the query is related to medical conditions, treatments, symptoms, or any other health-related topics, you should respond with **careful, evidence-based, and precise information**. If you're unsure about any aspect of the response, it is better to indicate that medical advice should be sought from a healthcare provider. Always prioritize the safety and well-being of the user.

#     If the user's query is **not related to any medical field**, respond with the message: 
#     "Sorry! I am a medical bot, I cannot answer general questions."

#     **Key instructions**:
#     - Ensure your response is grounded in factual, medical evidence and knowledge.
#     - Do **not** make up answers or give unverified information.
#     - If the question concerns medical symptoms or conditions, be **cautious and detailed**, but avoid diagnosing or prescribing. Always advise users to consult a medical professional for confirmation.
#     - If the query is about something non-medical, give a polite, clear, and firm redirection.
    
#     User Query: "{user_query}"
#     """
#     return prompt

# chatbot_icon_html = """
# <style>
#     .chatbot-icon {
#         position: fixed;
#         bottom: 20px;
#         right: 20px;
#         width: 60px;
#         height: 60px;
#         border-radius: 50%;
#         background-color: #4CAF50;
#         color: white;
#         display: flex;
#         justify-content: center;
#         align-items: center;
#         font-size: 24px;
#         box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
#         cursor: pointer;
#         z-index: 10;
#     }
# </style>
# <div onclick="document.getElementById('chatbot-button').click()" class="chatbot-icon">💬</div>
# """

# st.markdown(chatbot_icon_html, unsafe_allow_html=True)

selected=option_menu(None,['Home','MediScan','RecordDelve'],
    icons=['house','book','envelope'],
    menu_icon='cast',
    default_index=1,
    orientation='horizontal',
)

st.markdown(
    """
    <style>
    .cover-glow {
        width: 100%;
        height: auto;
        padding: 3px;
        box-shadow: 
            0 0 5px #330000,
            0 0 10px #660000,
            0 0 15px #990000,
            0 0 20px #CC0000,
            0 0 25px #FF0000,
            0 0 30px #FF3333,
            0 0 35px #FF6666;
        position: relative;
        z-index: -1;
        border-radius: 30px;  /* Rounded corners */
    }
    </style>
    """,
    unsafe_allow_html=True,
)
def img_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Load and display sidebar image with glowing effect
img_path = "Intellicare.png"
img_base64 = img_to_base64(img_path)
st.sidebar.markdown(
    f'<img src="data:image/png;base64,{img_base64}" class="cover-glow">',
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

st.sidebar.markdown("# Team Members")

st.sidebar.markdown("# :orange[Karthikeyan M]")
st.sidebar.markdown("# :orange[Arun Kumar R]")
st.sidebar.markdown("# :orange[Logabaalan R S]")
st.sidebar.markdown("# :orange[Barath Raj P]")

# st.sidebar.markdown(
#         """
        
#         Follow me on:

#         LinkedIn → [Barath Raj P](https://www.linkedin.com/in/barathrajp/)

#         """
#     )


# st.sidebar.markdown(
#         """
        
#         Follow me on:

#         LinkedIn → [Karthikeyan M](https://www.linkedin.com/in/karthikeyan-m30112004/)

#         """
#     )


# st.sidebar.markdown(
#         """
        
#         Follow me on:

#         LinkedIn → [Arun Kumar R](https://www.linkedin.com/in/arun-kumar-99b841255/)

#         """
#     )

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

res=[]
ques=[]
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def home():
    left_column, right_column = st.columns((2,5))

    with left_column:
        logo='https://lottie.host/49cfa049-139a-498a-954f-7985b2b60086/qvfWaOHQJR.json'
        logo_image=load_lottieurl(logo)
        st_lottie(logo_image,width=300,height=100,key='logo')
    with right_column:
        st.header("MediChat")

    selected_language = st.selectbox("Select your preferred language:", languages)

    if selected_language!='Select Your Language':
        selected_language_code = selected_language
        # def show():
        #     lottie_url = "https://lottie.host/00b6f013-63d8-4663-8829-edacfc48286d/v0PwepvHxL.json"
        #     hello = load_lottieurl(lottie_url)
        #     with st_lottie_spinner(hello,width=500, height=200,key="main"):
        #         time.sleep(5)

        tokenizer = AutoTokenizer.from_pretrained("K:\model")
        model = AutoModelForSeq2SeqLM.from_pretrained("K:\model")
        def generate_response(query):
            inputs = tokenizer.encode("answer: " + query, return_tensors="pt", max_length=512, truncation=True)
            outputs = model.generate(inputs, max_length=512, num_return_sequences=1)
            return tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if prompt :=  st.file_uploader("Upload a Prescription Image"):
            data = pd.read_csv('Medicine_Details.csv')
            file_bytes = prompt.read()
            nparr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            pipeline = keras_ocr.pipeline.Pipeline()
            value = pipeline.recognize([img])
            df = pd.DataFrame(value[0],columns=["Text",'Size'])
            ocr=[]
            for i in df['Text']:
                ocr.append(i)
            def search(prescription):
                ocr_med=pd.read_csv("ocr_lower.csv")
                extracted_medicines = []
                for word in prescription:
                    if word.lower() in ocr_med['Medicine'].values:
                        if len(word)>3:
                            extracted_medicines.append(word)
                print(prescription, extracted_medicines)
                return extracted_medicines
            result=search(ocr)
            medicine_name=result
            templates = {
                "usage": "What is {medicine} used for?",
                "contraindications": "Who should not take {medicine}?",
                "dosage": "How should I take {medicine}?",
                "general_info": "what are the side effects of {medicine}.",
                "Storage Query":"How should I store my {medicine}",
                "expire":"When does my prescription for {medicine} expire?",
                "man":"Who manufactures {medicine}",
                'overdose':'What are the symptoms of an {medicine} overdose, and what should I do if it happens?',
                "what":"What is {medicine}?",
                "before":"What should I tell my doctor before taking {medicine}?",
                "ingredients":"What are the ingredients in {medicine}?"
                }


            def generate_prompts(medicine_name):
                prompts = {}
                for key, template in templates.items():
                    prompts[key] = template.format(medicine=medicine_name)
                return prompts
            
            def geturl(q):
                api_key = "AIzaSyBdlN72Py11hJLSiflzzcp9MAZRv0e03pI"
                search_engine_id = "06e713d09dbd14dbc"
                query=f'{q} medicine uses,side effects and other details'
                url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={search_engine_id}&q={query}"
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    return data.get('items', [])
                else:
                    print("Error:", response.status_code)
                return None
            def translate(info,lang):
                translator=Translator()
                translation = translator.translate(info, dest=LANGUAGES[lang])
                return translation.text
                
            st.sidebar.markdown("---")
        # Example usage
            
            pro=[]
            placeholders=[]
            for med in medicine_name:
                with st.expander(translate(med,selected_language_code)):
                    prompts = generate_prompts(med)
                    # with st.chat_message("user"):
                    #     st.write(med)
                    pro.append(med)
                    with st.chat_message("assistant"):
                        for prompt_type, prompt_text in prompts.items():
                            response = generate_response(prompt_text)
                            if len(response.split(','))<20:
                                ques.append(prompt_text)
                                res.append(response)
                                question_placeholder = st.empty()
                                response_placeholder = st.empty()
                                question_placeholder.markdown(f"**❔Question:** {translate(prompt_text, selected_language_code)}")
                                response_placeholder.markdown(f"**🙋Answer:** {translate(response, selected_language_code)}")
                                # st.markdown(f"**Question:** {prompt_text}")              
                                # st.markdown(f"**Answer:** {response}")
                                placeholders.append((question_placeholder, response_placeholder))
                                st.write("")
                        st.write("-----------------------------------------------------------------")
                        # default_image='https://raw.githubusercontent.com/ArunKumar200510/MediChat-Anokha/main/images/empty.png'
                        # img=find_image_url(med,data)
                        # if img!=default_image:
                        #     st.image(img, use_column_width=True)
                        # else:
                        #     st.image(img,width=10)
                        url=geturl(med)
                        if(url is not None and len(url)>0):
                            st.write(f" - {url[0].get('link')}")
                            url=None
                
            # selected_option = st.selectbox("Select an option if you want to Translate",languages )
            # if(selected_option != 'Select an option if you want to Translate' ):
            #     for idx, (question_placeholder, response_placeholder) in enumerate(placeholders):
            #         translated_question = translate(ques[idx],selected_option )
            #         translated_response = translate(res[idx], selected_option)
            #         question_placeholder.markdown(f"**Question:** {translated_question}")
            #         response_placeholder.markdown(f"**Answer:** {translated_response}")
            #     selected_option = 'Select an option if you want to Translate'
            # st.warning('As an AI language model, I must clarify that I am not a healthcare professional, so I cannot provide medical advice. However, I can offer some general information.', icon="⚠️")

def extract_text_from_pdf(pdf_path): 
    pdf_bytes = BytesIO(pdf_path.read())
    pdf_reader = PyPDF2.PdfReader(pdf_bytes)
    page_num = 0
    page = pdf_reader.pages[page_num]
    text = page.extract_text()
    return text

# Helper function to convert images to base64 format
def img_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

# Helper function for displaying images with base64 and rounded corners
def image_card(base64_img, title=None):
    img_str = f"data:image/png;base64,{base64_img}"
    st.markdown(
        f"""
        <div style="display: flex; justify-content: center; padding: 20px;">
            <img src="{img_str}" alt="Image" style="border-radius: 10px; max-width: 100%; height: auto;">
            {f"<h3 style='text-align: center; color: #333;'>{title}</h3>" if title else ""}
        </div>
        """,
        unsafe_allow_html=True
    )

# Member card with name, LinkedIn, and email
def member_card(image_base64, name, linkedin_url, email):
    card = f"""
    <div style="display: flex; flex-direction: column; align-items: center; margin: 10px;">
        <div style="border-radius: 50%; overflow: hidden; width: 150px; height: 150px;">
            <img src="data:image/png;base64,{image_base64}" style="width: 100%; height: auto;">
        </div>
        <h3 style="margin: 10px 0 5px 0;">{name}</h3>
        <a href="{linkedin_url}" target="_blank" style="color: #0e76a8; text-decoration: none; margin-bottom: 5px;">LinkedIn</a>
        <a href="mailto:{email}" style="color: #d44638; text-decoration: none;">{email}</a>
    </div>
    """
    return card

# Main function for the About page
def about_page():
    st.markdown("<h1 style='text-align: center;'>Home</h1>", unsafe_allow_html=True)

    # Problem Section
    st.markdown(
        """
        <div style="padding: 20px; margin-bottom: 20px; border-radius: 10px; background-color: #f7f7f7; border-left: 4px solid #0073e6;">
            <h2 style="color: #0073e6;">Problem</h2>
            <p style="color: #333;">
                In today's world, many people struggle to understand the medical information contained in doctor 
                prescriptions and medical reports. Without adequate knowledge of medicine, patients often use prescribed 
                medications without proper understanding or clarification. This can lead to misuse of medications, resulting 
                in ineffective treatment or adverse health effects.
            </p>
            <p style="color: #333;">
                Furthermore, medical reports are comprehensive documents that contain detailed information about a patient's 
                condition. However, they are often written in complex medical jargon, making it difficult for patients to 
                extract the essential details they need, such as diagnoses, symptoms, and treatments. This lack of clarity 
                can prevent patients from fully understanding their health status and the necessary steps for their treatment 
                and recovery.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Solution Section
    st.markdown(
        """
        <div style="padding: 20px; margin-bottom: 20px; border-radius: 10px; background-color: #e6f2ff; border-left: 4px solid #0073e6;">
            <h2 style="color: #0073e6;">Solution</h2>
            <p style="color: #333;">
                We have developed a <strong>Prescription Scanner</strong> project designed to help patients better understand 
                the medications prescribed to them. This project involves several Gen AI techniques, such as fine-tuned models, 
                to achieve accurate and informative results.
            </p>
            <p style="color: #333;">
                Additionally, our <strong>Report Scanner</strong> project leverages advanced technologies to interpret and summarize 
                medical reports for patients. By combining OCR, sentence embedding, vector databases, and retrieval-augmented generation (RAG), 
                we deliver precise and insightful diagnoses, symptoms, and treatments based on the medical reports.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Architecture Section
    architecture_img_base64 = img_to_base64("architecture.png")
    st.markdown(
        """
        <div style="padding: 20px; margin-bottom: 20px; border-radius: 10px; background-color: #f0f8ff; border-left: 4px solid #0073e6;">
            <h2 style="color: #0073e6;">Architecture</h2>
        </div>
        """,
        unsafe_allow_html=True
    )
    image_card(architecture_img_base64)

    # Display developer cards in a row
    member1_img_base64 = img_to_base64("Karthi.jpeg")
    member2_img_base64 = img_to_base64("Arun.png")
    member3_img_base64 = img_to_base64("Barath.jpeg")
    member4_img_base64 = img_to_base64("Loga.jpg")

    member1_card = member_card(member1_img_base64, "Karthikeyan M", "https://www.linkedin.com/in/karthikeyan-m30112004/", "karthikeyanmjnk13579@gmail.com")
    member2_card = member_card(member2_img_base64, "Arun Kumar R", "https://www.linkedin.com/in/arun-kumar-99b841255/", "arun700101@gmail.com")
    member3_card = member_card(member3_img_base64, "Barath Raj P", "https://www.linkedin.com/in/barathrajp/", "prakasambarath289@gmail.com")
    member4_card = member_card(member4_img_base64, "Logabaalan R S", "https://www.linkedin.com/in/logabaalan-r-s-94ba82259/", "logabaalan2004@gmail.com")

    st.markdown(
        f"""
        <div style="display: flex; justify-content: center; flex-wrap: wrap;">
            {member1_card}
            {member2_card}
            {member3_card}
            {member4_card}
        """,
        unsafe_allow_html=True
    )

if selected=='MediScan':
    home()
if selected=='RecordDelve':
    report()
if selected=='Home':
    about_page()