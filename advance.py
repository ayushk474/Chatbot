import os
import faiss
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
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

# Enable Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load API Key
from secret_api_keys import huggingface_api_key
os.environ['HUGGINGFACEHUB_API_TOKEN'] = huggingface_api_key
if not huggingface_api_key:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN is not set!")

# MongoDB Connection
MONGO_URI = "mongodb+srv://ayushk47:A1234@assistant.w3yor.mongodb.net/?retryWrites=true&w=majority&appName=Assistant"
DB_NAME = "Task"
COLLECTION_NAME = "Task_data"
COMMENT_COLLECTION = "Comments"

# Initialize FastAPI
app = FastAPI()
client = AsyncIOMotorClient(MONGO_URI)
db = client[DB_NAME]
task_collection = db[COLLECTION_NAME]
comment_collection = db[COMMENT_COLLECTION]

# Request Models
class Task(BaseModel):
    title: str
    description: str
    status: str
    priority: str
    due_date: str

class Comment(BaseModel):
    task_id: str
    comment: str
    timestamp: datetime = datetime.utcnow()

class QueryRequest(BaseModel):
    query: str
    model: str

MODEL_MAP = {
    "LLaMA": "meta-llama/Meta-Llama-3-8B-Instruct",
    "Qwen": "Qwen/Qwen2.5-3B-Instruct",
    "Hugging Face": "HuggingFaceH4/zephyr-7b-beta",
}

# Task Routes
@app.post("/task/create")
async def create_task(task: Task):
    task_dict = task.dict()
    task_dict['due_date'] = datetime.strptime(task.due_date, "%Y-%m-%d")
    result = await task_collection.insert_one(task_dict)
    return {"message": "Task created", "task_id": str(result.inserted_id)}

@app.get("/task/list")
async def list_tasks():
    tasks = await task_collection.find().to_list(None)
    return {"tasks": tasks}

@app.delete("/task/delete/{task_id}")
async def delete_task(task_id: str):
    result = await task_collection.delete_one({"_id": task_id})
    return {"message": "Task deleted"} if result.deleted_count else HTTPException(status_code=404, detail="Task not found")

@app.post("/task/comment")
async def add_comment(comment: Comment):
    await comment_collection.insert_one(comment.dict())
    return {"message": "Comment added"}

# AI Chat Assistant
async def fetch_tasks() -> List[Dict[str, Any]]:
    tasks = await task_collection.find().to_list(None)
    return tasks if tasks else []

def process_text(tasks: List[Dict[str, Any]]):
    formatted_texts = [
        f"Title: {task.get('title', 'No Title')}\n"
        f"Description: {task.get('description', 'No Description')}\n"
        f"Status: {task.get('status', 'Unknown')}\n"
        f"Priority: {task.get('priority', 'None')}\n"
        f"Due Date: {task.get('due_date', 'N/A')}"
        for task in tasks
    ]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_text("\n\n".join(formatted_texts))
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    index = faiss.IndexFlatL2(768)
    vector_store = FAISS(embedding_function=embeddings.embed_query, index=index, docstore=InMemoryDocstore(), index_to_docstore_id={})
    vector_store.add_texts(texts)
    return vector_store

@retry(wait=wait_fixed(5), stop=stop_after_attempt(3))
def query_model(llm, retriever, query):
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    similar_docs = retriever.get_relevant_documents(query)
    relevant_info = "\n".join([doc.page_content for doc in similar_docs[:3]])
    prompt = f"Relevant tasks:\n{relevant_info}\n\nAnswer: {query}"
    response = qa({"query": prompt})
    return response.get("result", "No response generated")

@app.post("/chat")
async def chat(query_request: QueryRequest):
    tasks = await fetch_tasks()
    if not tasks:
        raise HTTPException(status_code=500, detail="No tasks found")
    if query_request.model not in MODEL_MAP:
        raise HTTPException(status_code=400, detail="Invalid model selection")
    vectorstore = process_text(tasks)
    retriever = vectorstore.as_retriever()
    llm = HuggingFaceEndpoint(repo_id=MODEL_MAP[query_request.model], token=huggingface_api_key, temperature=0.6)
    response = query_model(llm, retriever, query_request.query)
    return {"response": response}

# Notifications
@app.get("/task/reminders")
async def send_reminders():
    now = datetime.utcnow()
    upcoming_tasks = await task_collection.find({"due_date": {"$lte": now + timedelta(days=1)}}).to_list(None)
    return {"upcoming_tasks": upcoming_tasks}

# Run FastAPI
import uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)