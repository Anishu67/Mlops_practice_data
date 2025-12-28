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
            message = "Invalid token"
        )
    
app.post("/login")
def login(data : LoginRequest):
    if data.username == "admin" and data.password == "admin123":
        token = create_access_token({"sub": data.username})
        return {
            "access_token" : token,
            "message" : "You have logged in Successfully"
        }
    raise HTTPException(
        status_code= 401,
        message = "Invalid credentials"
    )

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

raw_text = load_pdf_text("sample_policy.pdf")
chunks = chunk_text(raw_text)


    

