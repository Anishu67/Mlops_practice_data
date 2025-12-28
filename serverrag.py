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
#rag initiation
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

raw_text = load_pdf_text("sample_policy.pdf")
chunks = chunk_text(raw_text)

embedding = embedding_model.encode(chunks)
embedding = np.array(embedding).astype("float32")

index = faiss.IndexFlatL2(embedding.shape[1])
index.add(embedding)

def ask_rag(question:str) ->str:
    query_embedding = embedding_model.encode([question])
    k = 3
    distance, indices = index.search(np.array(query_embedding).astype("float32"), k = k)
    retrieve_doc = [chunks[i] for i in indices[0]]
    context = "\n".join(retrieve_doc)

    prompt = f"""
You are a strict question answering system.

Rules:
- Use ONLY the information present in the context.
- Do NOT add extra explanations.
- Do NOT invent policies, companies, or laws.
- If the answer is present, respond in ONE short sentence.
- If the answer is NOT present, respond exactly with: I don't know.

context :
{context}

question :
{question}

"""
    
    response = requests.post(
         "http://localhost:11434/api/generate",
        json = {
            "model":"phi3:mini",
            "prompt":prompt,
            "stream" : False
        }

    )


    data = response.json()
    if "response" in data:
        return data["response"]
    else:
        raise HTTPException(status_code=500, detail=data)

@app.post("/chat")

def chat(
    data:ChatRequest,
    user : str = Depends(get_current_user)
):

    answer = ask_rag(data.question)

    return{
        "user": user,
        "question": data.question,
        "answer": answer

    }