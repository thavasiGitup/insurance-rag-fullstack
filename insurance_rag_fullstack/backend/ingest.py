
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
