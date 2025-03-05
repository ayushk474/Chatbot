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
from datetime import datetime

# Load Hugging Face API Key
try:
    from secret_api_keys import huggingface_api_key
    os.environ['HUGGINGFACEHUB_API_TOKEN'] = huggingface_api_key
except ImportError:
    print("Error: secret_api_keys.py not found or huggingface_api_key not defined.")
    huggingface_api_key = None

# FastAPI App
app = FastAPI()

# Request Model
class QueryRequest(BaseModel):
    query: str
    model: str  # Model name

# Supported Models
MODEL_MAP = {
    "LLaMA": "meta-llama/Meta-Llama-3-8B-Instruct",
    "Qwen": "Qwen/Qwen2.5-3B-Instruct",
    "Hugging Face": "HuggingFaceH4/zephyr-7b-beta",
}

TIMEOUT = 300

### 🚀 FUNCTION: Fetch Task Data ###
def fetch_eshway_tasks():
    """Load tasks from a local JSON file (or integrate with MongoDB)."""
    try:
        if not os.path.exists("task.json"):
            print("❌ task.json file not found!")
            return None
        with open("task.json", "r") as f:
            return json.load(f)  # Return raw task data (list of dictionaries)
    except json.JSONDecodeError as e:
        print("JSON Parse Error:", e)
        return None
    except Exception as e:
        print(f"Unexpected Error: {e}")
        return None

### 🚀 FUNCTION: Parse Task Data ###
def parse_task_data(data):
    """Convert JSON task data into a structured text format."""
    tasks = []
    for task in data:
        title = task.get("title", "No Title")
        description = task.get("description", "No Description")
        tasks.append(f"**{title}**\n{description}")
    return "\n\n".join(tasks)

### 🚀 FUNCTION: Process Text into FAISS Vector Store ###
def process_text(text):
    """Convert text into FAISS vector store for efficient retrieval."""
    if not text:
        return None
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_text(text)
    hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    sample_embedding = np.array(hf_embeddings.embed_query("sample text"))
    dimension = sample_embedding.shape[0]
    index = faiss.IndexFlatL2(dimension)
    vector_store = FAISS(
        embedding_function=hf_embeddings.embed_query,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    vector_store.add_texts(texts)
    return vector_store

### 🚀 FUNCTION: Query LLM Model ###
@retry(wait=wait_fixed(5), stop=stop_after_attempt(3))
def query_model(llm, retriever, query):
    """Use LLM with FAISS retriever to generate a response."""
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa({"query": query})

### 🏡 API Home Route ###
@app.get("/")
def home():
    return {"message": "Welcome to the AI-Powered Project Management Assistant!"}

### 🤖 API Chat Route ###
@app.post("/chat")
def chat(query_request: QueryRequest):
    """Process user queries and return AI-powered responses."""
    task_data = fetch_eshway_tasks()
    if not task_data:
        raise HTTPException(status_code=500, detail="Failed to fetch tasks")

    query = query_request.query.lower()

    # ✅ Fetch Pending Tasks
    if "pending" in query:
        pending_tasks = [task for task in task_data if task.get("status", "").lower() != "completed"]
        return {"response": {"pending_tasks": pending_tasks} if pending_tasks else "No pending tasks found."}

    # ✅ Fetch Completed Tasks
    if "completed" in query:
        completed_tasks = [task for task in task_data if task.get("status", "").lower() == "completed"]
        return {"response": {"completed_tasks": completed_tasks} if completed_tasks else "No completed tasks found."}

    # ✅ Fetch Priority Tasks
    if "priority" in query:
        priority_tasks = [task for task in task_data if task.get("priority", "").lower() == "high"]
        return {"response": {"priority_tasks": priority_tasks} if priority_tasks else "No high-priority tasks found."}

    # ✅ Fetch Overdue Tasks
    if "overdue" in query:
        now = datetime.now()
        overdue_tasks = [
            task for task in task_data if task.get("due_date") and
            datetime.fromisoformat(task["due_date"].replace("Z", "+00:00")) < now and
            task.get("status", "").lower() != "completed"
        ]
        return {"response": {"overdue_tasks": overdue_tasks} if overdue_tasks else "No overdue tasks found."}

    # ✅ Analyze Current Project
    if "analyze" in query or "project status" in query:
        total_tasks = len(task_data)
        completed_tasks = sum(1 for task in task_data if task.get("status", "").lower() == "completed")
        pending_tasks = total_tasks - completed_tasks
        return {
            "response": {
                "total_tasks": total_tasks,
                "completed_tasks": completed_tasks,
                "pending_tasks": pending_tasks,
                "progress": f"{(completed_tasks / total_tasks) * 100:.2f}%" if total_tasks > 0 else "0%"
            }
        }

    # ✅ Guide on Completing Pending Tasks
    if "how to complete" in query:
        return {"response": "Break down tasks into smaller steps, prioritize high-impact tasks, and allocate team members effectively."}

    # ✅ Default LLM Query Handling
    if query_request.model not in MODEL_MAP:
        raise HTTPException(status_code=400, detail="Invalid model selection")

    text_data = parse_task_data(task_data)
    vectorstore = process_text(text_data)
    if not vectorstore:
        raise HTTPException(status_code=500, detail="Failed to process task data")

    retriever = vectorstore.as_retriever()
    llm = HuggingFaceEndpoint(repo_id=MODEL_MAP["Hugging Face"], token=huggingface_api_key, temperature=0.6, timeout=TIMEOUT)
    response = query_model(llm, retriever, query_request.query)

    return {"response": response}

