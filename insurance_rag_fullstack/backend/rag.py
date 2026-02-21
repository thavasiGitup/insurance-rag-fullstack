
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
        context = "\n\n".join(self.retrieve(query))
        prompt = f"""
You are an insurance assistant.
Answer ONLY from the context.

Context:
{context}

Question:
{query}
"""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )

        return response.choices[0].message.content
