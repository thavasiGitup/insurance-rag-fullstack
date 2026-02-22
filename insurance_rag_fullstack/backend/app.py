import os
import json
import pickle
import faiss
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from azure.storage.blob import BlobServiceClient
import uvicorn

# ==============================
# ENV VARIABLES
# ==============================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
BLOB_CONTAINER_NAME = os.getenv("BLOB_CONTAINER_NAME")
BLOB_FILE_NAME = os.getenv("BLOB_FILE_NAME")

if not OPENAI_API_KEY:
    raise Exception("OPENAI_API_KEY not set")

if not AZURE_STORAGE_CONNECTION_STRING:
    raise Exception("AZURE_STORAGE_CONNECTION_STRING not set")

# ==============================
# CLIENTS
# ==============================

client = OpenAI(api_key=OPENAI_API_KEY)
blob_service_client = BlobServiceClient.from_connection_string(
    AZURE_STORAGE_CONNECTION_STRING
)

# ==============================
# FASTAPI
# ==============================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str

# ==============================
# LOAD DATA FROM AZURE BLOB
# ==============================

def load_data_from_blob():
    container_client = blob_service_client.get_container_client(BLOB_CONTAINER_NAME)
    blob_client = container_client.get_blob_client(BLOB_FILE_NAME)
    blob_data = blob_client.download_blob().readall()
    return json.loads(blob_data)

# ==============================
# EMBEDDING
# ==============================

def embed(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding, dtype="float32")

def flatten(customer):
    text = f"Customer ID: {customer['customer_id']}\n"
    text += f"Name: {customer['name']}\n"
    text += f"Age: {customer['age']}\n"
    text += f"Email: {customer['email']}\n"

    for p in customer["policies"]:
        text += f"Policy ID: {p['policy_id']}\n"
        text += f"Type: {p['policy_type']}\n"
        text += f"Coverage: {p['coverage_amount']}\n"
        text += f"Premium: {p['premium']}\n"
        text += f"Status: {p['status']}\n"

    return text

# ==============================
# BUILD INDEX ON STARTUP
# ==============================

print("Loading data from Azure Blob...")
data = load_data_from_blob()

documents = [flatten(c) for c in data]

print("Creating embeddings...")

embeddings = np.array([embed(doc) for doc in documents])

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print("FAISS index created successfully.")

# ==============================
# RETRIEVAL
# ==============================

def retrieve(query, k=5):
    vector = embed(query)
    D, I = index.search(np.array([vector]), k)
    return [documents[i] for i in I[0]]

# ==============================
# API ENDPOINT
# ==============================

@app.post("/ask")
def ask(query: Query):
    context = "\n\n".join(retrieve(query.question))

    prompt = f"""
You are an insurance assistant.
Answer ONLY using the provided context.

Context:
{context}

Question:
{query.question}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )

    return {"answer": response.choices[0].message.content}


# ==============================
# HEALTH CHECK
# ==============================

@app.get("/")
def health():
    return {"status": "Azure RAG running"}