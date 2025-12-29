from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from datetime import datetime, timedelta
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests

from pdf_loader import load_pdf_text
from chunker import chunk_text

app = FastAPI()
security = HTTPBearer()

#CONFIG
SECRET_KEY = "your_secret_key"
EXPIRE_IN_MINUTES = 30
ALGORITHM = "HS256"

# ================= GLOBAL RAG STATE =================
embedding_model = None
faiss_index = None
chunks = []


class LoginRequest(BaseModel):
    username : str
    password : str  

class ChatRequest(BaseModel):
    question:str

def create_access_token(data:dict):
    payload = data.copy()
    payload["exp"] = datetime.utcnow() + timedelta(minutes = EXPIRE_IN_MINUTES)
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(credentials : HTTPAuthorizationCredentials = Depends(security)):

    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except JWTError:
        raise HTTPException(
            status_code= 401,
            detail = "Invalid token"
        )
    
@app.post("/login")
def login(data : LoginRequest):
    if data.username == "admin" and data.password == "admin123":
        token = create_access_token({"sub": data.username})
        return {
            "access_token" : token,
            "message" : "You have logged in Successfully"
        }
    raise HTTPException(
        status_code= 401,
        detail = "Invalid credentials"

    )


# ================= STARTUP EVENT =================

@app.on_event("startup")
def startup_rag():
    global embedding_model, faiss_index, chunks
    
    try:
        print("🔄 Initializing RAG pipeline...")


 # Load embedding model
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Load & process PDF
        raw_text = load_pdf_text("sample_policy1.pdf")
        chunks = chunk_text(raw_text)

        # Create embeddings
        embeddings = embedding_model.encode(chunks)
        embeddings = np.array(embeddings).astype("float32")

        # Build FAISS index
        faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
        faiss_index.add(embeddings)

        print("✅ RAG pipeline initialized successfully")

    except Exception as e:
        print("❌ RAG initialization failed:", str(e))
        faiss_index = None
        embedding_model = None
        chunks = []

        
def ask_rag(question: str) -> str:
    if embedding_model is None or faiss_index is None:
        return "RAG system not ready"

    query_embedding = embedding_model.encode([question])
    query_embedding = np.array(query_embedding).astype("float32")

    k = 3
    _, indices = faiss_index.search(query_embedding, k)

    retrieved_docs = [chunks[i] for i in indices[0]]
    context = "\n".join(retrieved_docs)

    prompt = f"""
You are a strict question answering system.

Rules:
- Use ONLY the information present in the context.
- Do NOT guess or hallucinate.
- If answer is not present, say: I don't know.

Context:
{context}

Question:
{question}
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "phi3:mini",
            "prompt": prompt,
            "stream": False
        },
        timeout=30
    )

    return response.json().get("response", "I don't know")

# ================= CHAT API =================

@app.post("/chat")
def chat(
    data: ChatRequest,
    user: str = Depends(get_current_user)
):
    answer = ask_rag(data.question)
    return {
        "user": user,
        "question": data.question,
        "answer": answer
    }