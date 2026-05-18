\# 🔥 RAG AI Assistant



This is a Retrieval-Augmented Generation (RAG) based AI project that answers user questions using custom documents and AI models.



\---



\## 🚀 What This Project Does



\- Takes user questions

\- Searches relevant information from stored data

\- Uses FAISS vector database for fast retrieval

\- Sends context to an AI model (Flan-T5)

\- Generates final intelligent answer

\- Displays everything in a Streamlit web app



\---



\## 🧠 How It Works



User Question  

→ Embedding Conversion  

→ FAISS Vector Search  

→ Retrieve Similar Text Chunks  

→ Pass to LLM (Flan-T5)  

→ Generate Final Answer  



\---



\## 📁 Project Files



\- `ingest.py` → Prepares data and creates embeddings

\- `retriever.py` → Finds relevant documents using vector search

\- `app.py` → Streamlit UI for chatbot

\- `vectorstore/` → Stores FAISS index

\- `data/` → Input documents

\- `requirements.txt` → Required libraries



\---



\## ⚙️ How to Run



\### 1. Install dependencies

```bash

pip install -r requirements.txt

