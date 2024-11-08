# 🌟 **Intellicare: Automated Medical Information Extraction** 🌟

Welcome to **Intellicare** by Team **Drastic Innovators**! Our project aims to enhance healthcare by enabling easy access and understanding of complex medical information through advanced AI models. 🩺💊

## 📢 Project Overview

In today's world, patients often struggle to understand medical information in prescriptions and reports, leading to potential misuse and confusion. Our solution uses cutting-edge technology like **OCR**, **NER**, **custom Large Language Models (LLMs)**, and **Retrieval-Augmented Generation (RAG)** to simplify medical information for better patient outcomes.

---

## 🚀 Solution Components

### **1. Prescription Scanner** 📄💊
   - **OCR (Optical Character Recognition):** Converts printed prescriptions to digital text.
   - **NER (Named Entity Recognition):** Identifies and extracts medication names.
   - **Custom LLM:** Fine-tuned using medical data, this model provides detailed information on each medication, including dosage, usage, and potential side effects.

### **2. Report Scanner** 📑
   - **OCR:** Extracts text from printed medical reports.
   - **Text Chunking and Embedding:** Embeds and stores text chunks as high-dimensional vectors.
   - **Vector Database (Qdrant):** Efficient storage and retrieval for quick similarity searches.
   - **Retrieval-Augmented Generation (RAG):** Uses retrieved data to generate summaries and insights, helping patients understand complex medical reports.

---

## 📊 **Architecture Diagrams**

| **Prescription Scanner** | **Report Scanner** |
| ------------------------ | ------------------ |
| ![Prescription Scanner Architecture](images/prescription_scanner_architecture.png) | ![Report Scanner Architecture](images/report_scanner_architecture.png) |

---

## 💡 **Innovation**

Our project is unique in integrating these tools into a **chatbot** that guides patients with detailed explanations of medical terms and reports. This AI-powered approach provides:
- **Drug Identification and Explanations**: Descriptions, dosages, and side effects.
- **Comprehensive Report Summaries**: Concise, understandable summaries of medical conditions and treatments.

---

## 👥 Meet the Team

- **Karthikeyan M**
- **Arun Kumar R**
- **Logabaalan R S**

🎓 **Sri Eshwar College of Engineering**

---

## 📹 **Demo Video**

👉 [Watch our demo](link-to-demo) to see the project in action and learn how Intellicare can transform healthcare for patients.

---

## 🛠 **Tech Stack**

- **OCR Tools**: Keras OCR
- **Named Entity Recognition**: Custom NER model
- **Large Language Model (LLM)**: Fine-tuned using Google Flan-T5 Instruct v1.0
- **Vector Database**: Qdrant on AWS Cloud
- **Sentence Embedding**: Sentence Transformers for text embeddings
- **Framework**: Retrieval-Augmented Generation (RAG) for response generation

---

## 📜 **Problem Statement**

Patients lack clarity when interpreting complex prescriptions and medical reports, risking improper medication use and misunderstanding their health conditions. Our solution is a step towards clear, accessible healthcare information.

---

## 🛠 **Implementation**

- **Prescription Scanner** extracts medication details and provides explanations.
- **Report Scanner** interprets medical reports, embedding chunks and generating summaries for easy understanding.

---

## 🎉 **Get Started**

1. **Clone the Repository**: `git clone https://github.com/your-username/intellicare.git`
2. **Install Requirements**: `pip install -r requirements.txt`
3. **Run the Project**: `python main.py`

For detailed setup instructions, see the [Installation Guide](link-to-installation-guide).

---

🌐 **It's all about improving access to quality care**
