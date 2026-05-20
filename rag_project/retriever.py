from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

with open("data/notes.txt", "r", encoding="utf-8") as f:
    docs = [line.strip() for line in f if line.strip()]

embeddings = model.encode(docs)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

def retrieve(query, k=3):
    q_emb = model.encode([query])
    _, idx = index.search(np.array(q_emb), k)
    return [docs[i] for i in idx[0]]