import os
import json
import random
from pathlib import Path

PROJECT_NAME = "insurance_rag_fullstack"

# =========================
# DATA GENERATION
# =========================

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
                "currency": "USD",
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


# =========================
# BACKEND FILES
# =========================

BACKEND_MAIN = """
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from rag import RAG

load_dotenv()

app = FastAPI()
rag = RAG()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask(query: Query):
    answer = rag.ask(query.question)
    return {"answer": answer}
"""

BACKEND_RAG = """
import os
import faiss
import pickle
import numpy as np
from openai import OpenAI

class RAG:
    def __init__(self):
        self.client = OpenAI()
        self.index = faiss.read_index("../vectorstore/index.faiss")
        with open("../vectorstore/meta.pkl", "rb") as f:
            self.docs = pickle.load(f)

    def embed(self, text):
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return np.array(response.data[0].embedding, dtype="float32")

    def retrieve(self, query, k=5):
        vec = self.embed(query)
        D, I = self.index.search(np.array([vec]), k)
        return [self.docs[i] for i in I[0]]

    def ask(self, query):
        context = "\\n\\n".join(self.retrieve(query))
        prompt = f\"\"\"
You are an insurance assistant.
Answer ONLY from the context.

Context:
{context}

Question:
{query}
\"\"\"

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )

        return response.choices[0].message.content
"""

BACKEND_INGEST = """
import json
import faiss
import pickle
import numpy as np
from openai import OpenAI

client = OpenAI()

def embed(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding, dtype="float32")

def flatten(customer):
    text = f"Customer ID: {customer['customer_id']}\\n"
    text += f"Name: {customer['name']}\\n"
    text += f"Age: {customer['age']}\\n"
    text += f"Email: {customer['email']}\\n"

    for p in customer["policies"]:
        text += f"Policy ID: {p['policy_id']}\\n"
        text += f"Type: {p['policy_type']}\\n"
        text += f"Coverage: {p['coverage_amount']}\\n"
        text += f"Premium: {p['premium']}\\n"
        text += f"Status: {p['status']}\\n"

    return text

def main():
    with open("../data/customers.json") as f:
        data = json.load(f)

    docs = [flatten(c) for c in data]

    embeddings = np.array([embed(d) for d in docs])

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, "../vectorstore/index.faiss")

    with open("../vectorstore/meta.pkl", "wb") as f:
        pickle.dump(docs, f)

    print("Ingestion complete.")

if __name__ == "__main__":
    main()
"""

BACKEND_REQUIREMENTS = """
fastapi
uvicorn
openai
faiss-cpu
numpy
python-dotenv
"""

# =========================
# FRONTEND FILES
# =========================

FRONTEND_HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>Insurance RAG</title>
  <link rel="stylesheet" href="style.css">
</head>
<body>
  <div class="chat">
    <h2>Insurance Assistant</h2>
    <div id="messages"></div>
    <input id="input" placeholder="Ask about policies..." />
    <button onclick="send()">Send</button>
  </div>
<script src="script.js"></script>
</body>
</html>
"""

FRONTEND_JS = """
async function send() {
    const input = document.getElementById("input");
    const messages = document.getElementById("messages");

    const question = input.value;
    if (!question) return;

    messages.innerHTML += "<div class='user'>" + question + "</div>";
    input.value = "";

    const response = await fetch("http://127.0.0.1:8000/ask", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({question})
    });

    const data = await response.json();
    messages.innerHTML += "<div class='bot'>" + data.answer + "</div>";
}
"""

FRONTEND_CSS = """
body { font-family: Arial; background: #f0f0f0; }
.chat { width: 500px; margin: 40px auto; background: white; padding: 20px; border-radius: 8px; }
#messages { height: 300px; overflow-y: auto; border: 1px solid #ddd; margin-bottom: 10px; padding: 10px; }
.user { color: blue; margin-bottom: 5px; }
.bot { color: green; margin-bottom: 10px; }
"""

ENV_FILE = "OPENAI_API_KEY=your_openai_api_key_here"

README = """
SETUP:

1. python create_project.py
2. cd insurance_rag_fullstack
3. Create backend/.env and paste your OpenAI key
4. cd backend
5. pip install -r requirements.txt
6. python ingest.py
7. uvicorn main:app --reload
8. Open frontend/index.html
"""

# =========================
# CREATE PROJECT
# =========================

def create_project():
    base = Path(PROJECT_NAME)

    (base / "backend").mkdir(parents=True, exist_ok=True)
    (base / "frontend").mkdir(parents=True, exist_ok=True)
    (base / "data").mkdir(parents=True, exist_ok=True)
    (base / "vectorstore").mkdir(parents=True, exist_ok=True)

    (base / "backend/main.py").write_text(BACKEND_MAIN)
    (base / "backend/rag.py").write_text(BACKEND_RAG)
    (base / "backend/ingest.py").write_text(BACKEND_INGEST)
    (base / "backend/requirements.txt").write_text(BACKEND_REQUIREMENTS)

    (base / "frontend/index.html").write_text(FRONTEND_HTML)
    (base / "frontend/script.js").write_text(FRONTEND_JS)
    (base / "frontend/style.css").write_text(FRONTEND_CSS)

    data = generate_data(100)
    with open(base / "data/customers.json", "w") as f:
        json.dump(data, f, indent=2)

    (base / "README.md").write_text(README)

    print("Project created successfully.")

if __name__ == "__main__":
    create_project()