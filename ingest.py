from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

with open("data/notes.txt", "r", encoding="utf-8") as f:
    text = f.read()

chunks = text.split("\n")

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)

index = faiss.IndexFlatL2(len(embeddings[0]))
index.add(np.array(embeddings))

faiss.write_index(index, "vectorstore/index.faiss")
np.save("vectorstore/chunks.npy", chunks)

print("Ingestion complete")