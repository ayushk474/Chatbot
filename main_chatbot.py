import requests
import json
import faiss
import os
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEndpoint
from tenacity import retry, wait_fixed, stop_after_attempt

# Load Hugging Face API Key
from secret_api_keys import huggingface_api_key
os.environ['HUGGINGFACEHUB_API_TOKEN'] = huggingface_api_key

# API for Fetching Tasks
ESHWAY_TASKS_API = "https://ltd-app.onrender.com/v1/projects/124/tasks/?search="

# Model Mapping
MODEL_MAP = {
    "LLaMA": "meta-llama/Meta-Llama-3-8B-Instruct",
    "Qwen": "Qwen/Qwen2.5-3B-Instruct",
    "Hugging Face": "sentence-transformers/all-MiniLM-L6-v2",
}

TIMEOUT = 300

# FastAPI App
app = FastAPI()

# Request Model
class QueryRequest(BaseModel):
    query: str
    model: str

# Fetch Data from Project Management System
def fetch_eshway_tasks():
    try:
        headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
        response = requests.get(ESHWAY_TASKS_API, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        return parse_task_data(data)
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching tasks: {e}")
        return None

def parse_task_data(data):
    tasks = []
    if isinstance(data, dict) and "tasks" in data:
        for task in data["tasks"]:
            task_text = f"**{task.get('title', 'No Title')}**\n{task.get('description', 'No Description')}"
            tasks.append(task_text)
    return "\n\n".join(tasks)

# Process Data into FAISS VectorStore
def process_text(text):
    if not text:
        return None
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_text(text)
    hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    
    sample_embedding = np.array(hf_embeddings.embed_query("sample text"))
    dimension = sample_embedding.shape[0]

    index = faiss.IndexFlatL2(dimension)
    vector_store = FAISS(
        embedding_function=lambda x: np.array(hf_embeddings.embed_query(x)),
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    vector_store.add_texts(texts)
    return vector_store

# AI Model Query with Retry
@retry(wait=wait_fixed(5), stop=stop_after_attempt(3))
def query_model(llm, retriever, query):
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa({"query": query})

# API Endpoint to Handle Chat Queries
@app.post("/chat")
def chat(query_request: QueryRequest):
    task_data = fetch_eshway_tasks()
    if not task_data:
        raise HTTPException(status_code=500, detail="Failed to fetch tasks")

    vectorstore = process_text(task_data)
    if not vectorstore:
        raise HTTPException(status_code=500, detail="Failed to process task data")

    if query_request.model not in MODEL_MAP:
        raise HTTPException(status_code=400, detail="Invalid model selection")

    llm = HuggingFaceEndpoint(
        repo_id=MODEL_MAP[query_request.model],
        token=huggingface_api_key,
        temperature=0.6,
        timeout=TIMEOUT,
    )

    response = query_model(llm, vectorstore.as_retriever(), query_request.query)
    return {"response": response}
