from flask import Flask, request, jsonify
import os
import json
from qdrant_client import QdrantClient, models
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import VectorParams, Distance
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import OpenAIEmbeddings
from huggingface_hub import InferenceClient
from qdrant_client.http.exceptions import UnexpectedResponse
import PyPDF2
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Initialize global objects
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
client = QdrantClient(
    url="https://2213f450-22de-421d-904f-1f69377c3137.us-east4-0.gcp.cloud.qdrant.io:6333", 
    api_key="YOUR_API",
)

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return [self.model.encode(text) for text in texts]

    def embed_query(self, text):
        return self.model.encode(text)

embedding_model_instance = SentenceTransformerEmbeddings()

def generate_embedding(text):
    try:
        return embedding_model.encode(text)
    except Exception as e:
        print(f"Error generating embedding: {e}")
        raise

def generate_embeddings(text):
    chunks = chunk_text(text)
    embeddings = [generate_embedding(chunk) for chunk in chunks]
    return embeddings, chunks

def chunk_text(text, chunk_size=1000, overlap=100):
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)

def generate_embedding_for_question(question):
    try:
        return embedding_model.encode([question])[0]
    except Exception as e:
        raise Exception(f"Error generating embedding: {str(e)}")

def fetch_relevant_context(user_id, collection_id, question):
    try:
        question_embedding = generate_embedding_for_question(question)
        collection_name = f"{user_id}_{collection_id}"
        search_result = client.query_points(
            collection_name=collection_name,
            query=question_embedding,
            limit=2,
        )
        if search_result:
            res1 = [resda for resda in search_result]
            res = [result[1] for result in res1]
            return res
        return None
    except Exception as e:
        raise Exception(f"Error fetching relevant context: {str(e)}")


def get_ans(context, ques):
    try:
        prompt=(f"You are an advanced AI assistant. Your task is to provide accurate and relevant responses based on the given context. "
                f"You should use only the provided context to answer the user's query. Do not deviate from the topic."
                f"Only use the given Context to answer the Question given by the user."
                f"#Note#: Do not generate answers on assumption. If you cannot find relevent answer from the Context just reply as 'No Relevent Context Found in the uploaded file'"
                f"#Context#: {context}"
                f"#Question#: {ques}"
                )
        client = InferenceClient(api_key="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")

        messages = [{"role": "user", "content": prompt}]

        stream = client.chat.completions.create(
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            messages=messages,
            max_tokens=500,
            stream=True
        )

        res = ""
        for chunk in stream:
            res += chunk.choices[0].delta.content  

        return res
            
    except Exception as e:
        raise Exception(f"Error querying OpenAI API: {str(e)}")

def create_vectorstore(user_id, collection_id):
    try:
        sample_embedding = embedding_model_instance.embed_documents(["Hi"])[0]
        vector_size = len(sample_embedding)
        
        collection_name = f"{user_id}_{collection_id}"
        
        if not collection_exists_safe(client, collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            )
            print("Collection created successfully")
        else:
            print("Collection already exists")
        
        vectorstore = QdrantVectorStore(client=client, collection_name=collection_name, embedding=embedding_model_instance)
        return vectorstore
    except Exception as e:
        print(f"Error creating vectorstore: {e}")
        raise

def store_embeddings_in_qdrant(vectorstore, embeddings, chunks, metadata):
    payloads = [
        {**metadata, 'chunk_index': i, 'text': chunk} 
        for i, (embedding, chunk) in enumerate(zip(embeddings, chunks))
    ]
    
    vectorstore.client.upload_collection(
        collection_name=vectorstore.collection_name,
        vectors=embeddings,
        payload=payloads
    )

def collection_exists_safe(client, collection_name):
    try:
        collections_response = client.get_collections()
        collection_names = [col.name for col in collections_response.collections]
        return collection_name in collection_names
    except Exception as e:
        print(f"Error checking collection existence: {e}")
        return False

@app.route('/process_pdf', methods=['POST'])
def process_pdf():
    data = request.get_json()
    text = data['text']
    user_id = data['user_id']
    collection_id = data['collection_id']
    
    try:
        embeddings, chunks = generate_embeddings(text)
        vectorstore = create_vectorstore(user_id, collection_id)
        metadata = {"collection_id": collection_id, "user_id": user_id}
        store_embeddings_in_qdrant(vectorstore, embeddings, chunks, metadata)
        return jsonify({"message": "PDF embeddings generated and stored successfully."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/delete_collection', methods=['POST'])
def delete_collection():
    data = request.get_json()
    user_id = data['user_id']
    collection_id = data['collection_id']
    collection_name = f"{user_id}_{collection_id}"
    
    try:
        client.delete_collection(collection_name=collection_name)
        return jsonify({"message": f"Collection '{collection_name}' deleted successfully."}), 200
    except UnexpectedResponse as e:
        return jsonify({"error": f"Failed to delete collection '{collection_name}': {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route('/rag_implement_retrive', methods=['POST'])
def rag_implement_retrieve():
    data = request.get_json()
    user_id = data['user_id']
    collection_id = data['collection_id']
    question = data['question']
    
    try:
        print(user_id, collection_id, question)
        relevant_context = fetch_relevant_context(user_id, collection_id, question)
        if not relevant_context:
            return jsonify({"message": "No relevant context found."}), 404

        response_message = get_ans(relevant_context, question)
        return jsonify({"response": response_message}), 200
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
