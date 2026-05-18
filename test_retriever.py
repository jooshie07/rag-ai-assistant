from retriever import retrieve

query = "What is FAISS?"
results = retrieve(query)

print("Results:\n")
for r in results:
    print("-", r)