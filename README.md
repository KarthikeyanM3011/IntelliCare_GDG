# Team Name : **Drastic Innovators**
# 🤖 INTELLICARE  : *Automated Medical Information Extraction and Patient Query Resolution Using OCR, NER, Fine-Tuned Models, and Retrieval-Augmented Generation (RAG)📑*

Welcome to **Intellicare** by Team **Drastic Innovators**! Our project aims to enhance healthcare by enabling easy access and understanding of complex medical information through advanced AI models. 🩺💊

## 📢 **Problem Statement**

In today's world, many people face significant challenges in understanding the medical information contained in doctor prescriptions and medical reports. These documents, though critical for effective healthcare, are often difficult to interpret due to complex medical jargon and unclear explanations. This lack of clarity can result in various issues, including:

### 💊 **Misuse of Medications**:

Patients often rely on prescriptions without fully understanding the medication instructions, dosages, or potential side effects. Without proper knowledge, patients might misuse medications, which can lead to ineffective treatment or even adverse health effects.

### 📄 **Complex Medical Reports**: 

Medical reports are often lengthy, filled with technical terms, and difficult to comprehend for individuals without a medical background. Important information, such as diagnoses, symptoms, and recommended treatments, may be hidden within these complex documents, preventing patients from fully understanding their health status or the steps they need to take for treatment and recovery.

---
# 🚀 Solutions
---
### 💊 **Prescription Scanner: Medication Understanding Simplified**

Our **Prescription Scanner** project is designed to help patients better understand the medications prescribed to them by leveraging **Generative AI (Gen AI)** techniques, such as:

- 🤖 **Fine-Tuned Model**: A custom fine-tuned model tailored to interpret prescription details accurately.
  
- 🧠 **AI-Powered Insights**: The system provides clear, understandable information about the medications, including dosage, potential side effects, and usage instructions.
  
This tool empowers patients to make informed decisions, improving their ability to follow treatment plans and reducing the risks of medication misuse or adverse effects. The AI system also ensures that the interpretation is simple and accessible to individuals without a medical background.

---

### 📄 **Report Scanner: Simplifying Medical Report Understanding**

Our **Report Scanner** project leverages cutting-edge technologies to interpret and summarize complex medical reports, making it easier for patients to understand their health status. Key features include:

- 📸 **OCR (Optical Character Recognition)**: Extracts text from medical reports, whether handwritten or typed.
  
- 🧑‍⚕️ **Sentence Embedding**: Uses deep learning models to understand the context and extract meaningful insights from medical reports.
  
- 📚 **Vector Databases**: Stores extracted information in vectorized form for fast and accurate retrieval.
  
- 🔄 **Retrieval-Augmented Generation (RAG)**: Combines the power of retrieval and generative models to provide precise and insightful summaries, including diagnoses, symptoms, treatments, and next steps.

This solution helps patients comprehend their medical condition by breaking down complex medical jargon, ensuring they fully understand their diagnoses, symptoms, and recommended treatments. It empowers patients to take control of their health with clarity.

---

### 📄 **Hash-Based Storage System: Securely Storing Patient Records**

Our Hash-Based Storage System ensures that patient medical records are stored securely, and transparently. This approach leverages hash technology to provide a decentralized, tamper-proof, and patient-centric storage solution. Key features include:

- 🔒 **Decentralized Storage**: Stores patient data across a distributed network, ensuring that records are protected against data breaches and unauthorized access.
 
- 👤 **Patient-Centric Control**: Empowers patients with control over their records, allowing them to grant access permissions to healthcare providers as needed.

This system transforms traditional data storage in healthcare, enhancing patient privacy and data security, while giving patients more control over their personal health information.

---

## 📊 **Architecture Diagrams**

## **💊 Prescription Scanner Architecture**
![Pres](Images/Prescription_architechture.png)

### Prescription Scanner - Architecture Overview

### 1. User Interaction
   - 🌐 **User Selects Language**: Users can choose their preferred language for processing.
   - 📄 **File Upload**: Users upload a medical prescription in various formats (PDF, image, or document).

### 2. Text Extraction
   - 🔍 **OCR Processing**: Text and content from the uploaded document are extracted using [**Keras OCR**](https://keras-ocr.readthedocs.io/en/latest/).
   - 💊 **Medicine Dataset**: Medication-related data is extracted from trusted sources like [NLM](https://pubmed.ncbi.nlm.nih.gov/) (National Library of Medicine).

### 3. Named Entity Recognition (NER)
   - 📊 **NER Model Processing**: The extracted text is passed to a **NER model** to identify and extract specific details related to medicines.
   - 📝 **Data Extraction**: Extracted information is then structured and prepared for further analysis.

### 4. Data Collection and Fine-Tuning
   - 🗄️ **Data Generation**: Over 10,000 datapoints related to medication are generated [Kaggle](https://www.kaggle.com/datasets/shudhanshusingh/az-medicine-dataset-of-india) and [medical archives](https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/), used to fine-tune the model.
   - 🔠 **Tokenization**: The generated data points are tokenized for model training.
   - 🔧 **Model Fine-Tuning**: **LoRA** (Low-Rank Adaptation) is used to fine-tune the [**FLAN-T5**](https://huggingface.co/google/flan-t5-xxl) model with the prepared dataset.

### 5. Model Deployment
   - ☁️ **Hugging Face Deployment**: The fine-tuned [model](https://huggingface.co/Karthikeyan-M3011/medflan-t5-large) is deployed on Hugging Face for easy access and scalability.

### 6. Output
   - 🤖 **Information Extraction**: The fine-tuned model analyzes the prescription and produces valuable medication details.

---

## **📄 Report Scanner Architecture**
![Repr](Images/Report_architechture.png)

### Report Scanner - Architecture Overview

### 1. User Interaction
   - 🌐 **User Selects Language**: Users select their preferred language.
   - 📄 **File Upload**: Users upload a medical report in PDF or DOCX format.

### 2. Content Extraction
   - 🛠️ **File Parsing**: 
     - For PDFs, [**PyPDF2**](https://pypdf2.readthedocs.io/en/3.x/) is used to extract text.
     - For DOCX files, [**Python-docx**](https://python-docx.readthedocs.io/en/latest/) is utilized for extraction.
   - 📄 **Content Extraction**: Extracted content from the report is processed.

### 3. Content Enhancement
   - 🧠 **Mixtral 7B Model**: The extracted content is enhanced for more accurate retrieval using the [**Mixtral 7B**](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) model, which improves the text content for better query response.

### 4. Text Chunking and Embedding
   - 📑 **Text Chunking**: The enhanced content is broken down into manageable chunks for efficient processing.
   - 🧬 **Embedding with All-Mini-L6**: Each chunk is converted to a vector embedding using the [**All-Mini-L6**](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) model, making it compatible for vector search.

### 5. Vector Database Storage
   - 🗄️ **Vector Storage in Qdrant**: The vector embeddings are stored in a [**Qdrant**](https://qdrant.tech/) vector database, enabling quick retrieval based on cosine similarity.
) 
### 6. Query Processing and Retrieval
   - 🔍 **User Query**: The user inputs a query to search the report.
   - 🔢 **Query Vectorization**: The query is converted into a vector representation.
   - 🔝 **Top-K Selection**: Using cosine similarity, the system identifies the Top-K most relevant chunks related to the query.

### 7. Result Presentation
   - 📊 **Relevant Information Display**: The system retrieves and displays the most relevant sections of the report, addressing the user's query effectively.

---

## **📄Hash-Based Storage System Architecture**
![Med-Id](https://github.com/user-attachments/assets/9aaa987d-f650-4c48-9bdf-f285a578314d)

### Med-ID Database System Overview

### 👤 User Registration
- **Profile Setup**: User provides name, mobile number, and email address.
- **ID Generation**: Backend creates a unique User ID and sends it via SMS for verification.

### 🏥 Hospital Sync
- **Data Matching**: User details (name, mobile) are matched with hospital records in the backend.

### 📄 Document Upload
- **Upload Files**: User uploads medical documents, which are stored and linked to their User ID.

### 💾 Database Storage
- **Secure Storage**: User data and documents are stored in the database, accessible via the mobile number.
  
---

# 🔧⚙️Tech Stack

---

## 🖥️ Frontend
- [Streamlit](https://streamlit.io/) - A framework for creating beautiful, interactive web applications in Python.
  
---

## 🔧 Backend
- [Python](https://www.python.org/) - The core programming language used in this project.
- [Flask](https://flask.palletsprojects.com/) - A lightweight WSGI web application framework used for building the API.
- [Hugging Face](https://huggingface.co/) - A platform for machine learning models, used for model deployment and inference.
- [Mistral 7B](https://mistral.ai/) - A powerful language model used to enhance the quality of text processing.
- [Google FLAN-T5](https://ai.googleblog.com/2022/10/flan-t5-open-sourcing-largest.html) - A language model by Google, fine-tuned to handle natural language tasks.
- [Sentence Transformers](https://www.sbert.net/) - Used to convert text into embeddings for similarity searches.
- [Twilio](https://www.twilio.com/) - A communication platform to enable SMS, voice, and video integration.
- [Keras OCR](https://github.com/faustomorales/keras-ocr) - A library for OCR (Optical Character Recognition) in images and PDFs.
- [Named Entity Recognition (NER)](https://en.wikipedia.org/wiki/Named-entity_recognition) - A process to identify and classify entities in text data.

---

## 🗃️ Database
- [Qdrant](https://qdrant.tech/) - A vector database for storing and searching embeddings, optimized for similarity search.
- [Firebase](https://firebase.google.com/) - A platform by Google for real-time databases, authentication, and more.
- [SQLite3](https://www.sqlite.org/index.html) - A lightweight, disk-based database used for local storage.

---

### **Repository Structure**

This repository contains all the necessary files and directories to run and experiment with the various components of the project. Below is a detailed description of the folder and file structure:

---

### **Folders**

#### 1. [**Image**](Images)
- **Description:** This folder contains all images and architecture diagrams related to the project. These images are used for visualizing the workflow, models, and any related concepts to better understand the project's design and functionality.

#### 2. [**Dataset**](Dataset)
- **Description:** This folder contains all the data necessary for training, fine-tuning, and testing the machine learning models.

  - **`metadata.csv`**: Contains detailed information related to medicines, which is used for fine-tuning the Google Flan-T5 model.
  - **`medicine_details`**: A file containing a list of all medicine names and related details. This data is crucial for model training and data analysis.
  - **`ocr_lower`**: Modified version of the medicine names used for fine-tuning the Named Entity Recognition (NER) model.


### **Files**

#### 3. [**`Finetune.ipynb`**](Finetune.ipynb)
- **Description:** This Jupyter Notebook contains the code to fine-tune the Google Flan-T5 model. It includes data preprocessing, model setup, training loops, and evaluation code for customizing the Flan-T5 model on medicine-related data.

#### 4. [**`mediq.py`**](mediq.py)
- **Description:** This Python file contains the logic for navigating various pages of the project, including the home page and prescription scanner functionality. It handles the navigation and page transitions for the web app.

#### 5. [**`qdrant.py`**](qdrant.py)
- **Description:** This script is responsible for connecting to the Qdrant database for retrieval, storing, and indexing information. It serves as the interface between the machine learning model and the database, ensuring efficient data handling.

#### 6. [**`record.py`**](record.py)
- **Description:** This Python file handles the working of the report scanner. It also integrates with `qdrant.py` for calling the Qdrant database and utilizes a RAG (retrieval-augmented generation) model to process scanned data and generate responses.

#### 7. [**`store_web3.py`**](store_web3.py)
- **Description:** This file handles the operation of secured record uploads and downloads between patients and doctors. Using Web3 technology, it ensures the integrity and privacy of medical records uploaded or downloaded by authorized users.

---


### **Steps to Run the Project Locally**

Follow the steps below to set up and run the project on your local machine.

#### 1. Clone the Repository

First, clone the repository using Git:

```bash
git clone https://github.com/KarthikeyanM3011/IntelliCare_GDG.git
cd IntelliCare_GDG
```

#### 2. Install Required Dependencies

Navigate to the project directory and install the required dependencies from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

#### 3. Set Up API Tokens

You need to provide the following API keys and tokens for the project to function properly. You can add these values as global

- **Twilio**:
  - `TWILIO_AUTH_TOKEN`
  - `TWILIO_SID`
  
- **Hugging Face**:
  - `HUGGING_FACE_API_KEY` (for inference API)

- **Google**:
  - `GOOGLE_TRANSLATE_API_KEY` (for translation services)
  - `GOOGLE_SEARCH_ENGINE_API_KEY` (for Google Custom Search Engine)

- **Qdrant**:
  - `QDRANT_AUTH_TOKEN` (for connecting to the Qdrant database)

- **Firebase**:
  - `FIREBASE_AUTH_TOKEN` (for Firebase authentication)
.

#### 4. Run the Application

After setting up the environment and API tokens, run the Streamlit app for the `mediq.py` file (which contains the prescription scanner and navigation system):

```bash
streamlit run mediq.py
```

This will start a local server, and you can access the app in your browser.

#### 5. Run the Qdrant Integration

Next, run the `qdrant.py` script to start the Qdrant database integration. This will set up the connection for data storage, indexing, and retrieval:

```bash
python qdrant.py
```

---

**Note:** Ensure that all required tokens and keys are set up correctly before running the scripts to avoid any issues with API connectivity.

---

Feel free to adjust or add any other specific setup instructions based on your repository’s configuration.

🌐 **It's all about improving access to quality care**
