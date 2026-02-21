# Insurance RAG Fullstack

A full-stack Retrieval-Augmented Generation (RAG) system for querying insurance customer and policy data.

## 🚀 Features

- FastAPI backend
- FAISS vector search
- OpenAI embeddings
- Chat-based frontend
- Local JSON dataset
- REST API endpoint

## 🏗 Architecture

Frontend → FastAPI → Embedding → FAISS → Context → LLM

## ⚙️ Setup

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python ingest.py
uvicorn main:app --reload
