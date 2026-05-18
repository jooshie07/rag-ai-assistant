from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index("vectorstore/index.faiss")
chunks = np.load("vectorstore/chunks.npy", allow_pickle=True)

def retrieve(query):
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k=2)
    return [chunks[i] for i in I[0]]