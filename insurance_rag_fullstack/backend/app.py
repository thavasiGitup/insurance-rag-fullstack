import os
import json
import random
import pickle
import faiss
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
import uvicorn

# =============================
# LOAD ENV
# =============================

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise Exception("❌ OPENAI_API_KEY not found in .env file")

client = OpenAI()

# =============================
# GENERATE SAMPLE DATA
# =============================

DATA_FILE = "customers.json"
INDEX_FILE = "index.faiss"
META_FILE = "meta.pkl"

def generate_data(n=100):
    policy_types = ["Health", "Auto", "Home", "Life", "Travel"]
    statuses = ["Active", "Expired", "Cancelled", "Pending"]

    customers = []

    for i in range(1, n + 1):
        policies = []
        for j in range(random.randint(1, 3)):
            policies.append({
                "policy_id": f"POL-{i}-{j}",
                "policy_type": random.choice(policy_types),
                "coverage_amount": random.randint(10000, 500000),
                "premium": random.randint(200, 2000),
                "status": random.choice(statuses)
            })

        customers.append({
            "customer_id": f"CUST-{i}",
            "name": f"Customer {i}",
            "age": random.randint(21, 75),
            "email": f"customer{i}@mail.com",
            "policies": policies
        })

    return customers

if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, "w") as f:
        json.dump(generate_data(100), f, indent=2)

# =============================
# EMBEDDING
# =============================

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

# =============================
# CREATE INDEX (FIRST RUN)
# =============================

if not os.path.exists(INDEX_FILE):

    print("Creating embeddings and FAISS index...")

    with open(DATA_FILE) as f:
        data = json.load(f)

    documents = [flatten(c) for c in data]
    embeddings = np.array([embed(doc) for doc in documents])

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, INDEX_FILE)

    with open(META_FILE, "wb") as f:
        pickle.dump(documents, f)

    print("Index created successfully.")

# =============================
# LOAD INDEX
# =============================

index = faiss.read_index(INDEX_FILE)

with open(META_FILE, "rb") as f:
    documents = pickle.load(f)

# =============================
# FASTAPI APP
# =============================

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

def retrieve(query, k=5):
    vec = embed(query)
    D, I = index.search(np.array([vec]), k)
    return [documents[i] for i in I[0]]

@app.post("/ask")
def ask(query: Query):
    context = "\n\n".join(retrieve(query.question))

    prompt = f"""
You are an insurance assistant.
Answer ONLY from the context below.

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

# =============================
# SIMPLE FRONTEND (SERVED FROM BACKEND)
# =============================

@app.get("/", response_class=HTMLResponse)
def homepage():
    return """
    <html>
    <head>
        <title>Insurance RAG</title>
        <style>
            body { font