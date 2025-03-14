import os
import faiss
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEndpoint
from tenacity import retry, wait_fixed, stop_after_attempt
import logging

# 🔹 Enable Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Hugging Face API Key
huggingface_api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not huggingface_api_key:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN is not set in environment variables")
    
# 🔹 MongoDB Connection
MONGO_URI = "mongodb+srv://ayushk47:A1234@assistant.w3yor.mongodb.net/?retryWrites=true&w=majority&appName=Assistant"
DB_NAME = "Task"
COLLECTION_NAME = "Task_data"

# 🔹 Initialize FastAPI
app = FastAPI()

# 🔹 Initialize MongoDB Client
client = AsyncIOMotorClient(MONGO_URI)
db = client[DB_NAME]
task_collection = db[COLLECTION_NAME]

# 🔹 Request Model
class QueryRequest(BaseModel):
    query: str
    model: str

# 🔹 Model Mapping
MODEL_MAP = {
    "LLaMA": "meta-llama/Meta-Llama-3-8B-Instruct",
    "Qwen": "Qwen/Qwen2.5-3B-Instruct",
    "Hugging Face": "HuggingFaceH4/zephyr-7b-beta",
}


# FUNCTION: Fetch Tasks from MongoDB
async def fetch_eshway_tasks() -> List[Dict[str, Any]]:
    """Retrieve tasks from MongoDB collection."""
    try:
        tasks = await task_collection.find().to_list(None)
        if not tasks:
            logger.warning("No tasks found in MongoDB.")
        return tasks
    except Exception as e:
        logger.error(f"MongoDB fetch error: {e}")
        return []


# FUNCTION: Process Text into FAISS Vector Store
def process_text(tasks: List[Dict[str, Any]]):
    """Convert task descriptions into embeddings and store in FAISS index."""
    if not tasks:
        logger.warning("No tasks found to process.")
        return None

    # 🔹 Format task data
    formatted_texts = [
        f"Title: {task.get('title', 'No Title')}\n"
        f"Description: {task.get('description', 'No Description')}\n"
        f"Status: {task.get('status', 'Unknown')}\n"
        f"Priority: {task.get('priority', 'None')}\n"
        f"Due Date: {task.get('due_date', 'N/A')}"
        for task in tasks
    ]
    full_text = "\n\n".join(formatted_texts)

    # 🔹 Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_text(full_text)

    # 🔹 Generate embeddings and store in FAISS
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    index = faiss.IndexFlatL2(768)
    vector_store = FAISS(
        embedding_function=embeddings.embed_query,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    vector_store.add_texts(texts)

    logger.info(f"FAISS VectorStore contains {len(texts)} embedded texts.")
    return vector_store

# FUNCTION: Query LLM Model

@retry(wait=wait_fixed(5), stop=stop_after_attempt(3))
def query_model(llm, retriever, query):
    """Query the LLM using FAISS retriever and provide task context."""
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # 🔹 Retrieve relevant task descriptions
    similar_docs = retriever.get_relevant_documents(query)
    relevant_info = "\n".join([doc.page_content for doc in similar_docs[:3]])

    if not relevant_info:
        relevant_info = "No relevant task data found."

    # 🔹 Optimized query prompt
    prompt = (
        f"Here are some relevant project tasks:\n{relevant_info}\n\n"
        f"Based on this information, answer the query: {query}"
    )

    response = qa({"query": prompt})

    logger.info(f"Raw LLM Response: {response}")

    return response.get("result", "No response generated")

# API Home Route
@app.get("/")
def home():
    return {"message": "Welcome to Assistant of AI-Driven Project Management"}

# 🛠 API Debug Route - Check MongoDB Tasks
@app.get("/debug/tasks")
async def debug_tasks():
    """Retrieve all tasks from MongoDB for debugging."""
    tasks = await fetch_eshway_tasks()
    return {"tasks": tasks}

# API Chat Route
@app.post("/chat")
async def chat(query_request: QueryRequest):
    """Process user queries with AI-powered responses."""
    
    # 🔹 Fetch task data
    task_data = await fetch_eshway_tasks()
    if not task_data:
        raise HTTPException(status_code=500, detail="Failed to fetch tasks")

    # 🔹 Ensure valid model selection
    if query_request.model not in MODEL_MAP:
        logger.error(f"Invalid model selection: {query_request.model}")
        raise HTTPException(status_code=400, detail="Invalid model selection")

    logger.info(f"Using model: {query_request.model}")

    # 🔹 Process text for vector search
    vectorstore = process_text(task_data)
    if not vectorstore:
        raise HTTPException(status_code=500, detail="Failed to process task data")

    retriever = vectorstore.as_retriever()

    # 🔹 Initialize LLM
    llm = HuggingFaceEndpoint(
        repo_id=MODEL_MAP[query_request.model], 
        token=huggingface_api_key,
        temperature=0.6
    )

    # Query the model
    response = query_model(llm, retriever, query_request.query)

    return {"response": response}


# Run FastAPI

import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
