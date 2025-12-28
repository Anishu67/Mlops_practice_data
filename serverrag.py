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



