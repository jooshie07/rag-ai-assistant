import streamlit as st
from retriever import retrieve
from transformers import pipeline

st.set_page_config(
    page_title="RAG AI Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 RAG AI Assistant")

# Load model ONCE
@st.cache_resource
def load_llm():
    return pipeline("text-generation", model="google/flan-t5-base")

llm = load_llm()   # ✅ IMPORTANT: must be defined BEFORE using it

query = st.text_input("Ask something")

if st.button("Search"):
    if query:

        results = retrieve(query)
        context = "\n".join(results)

        prompt = f"""
You are a helpful AI assistant.

Use the context below to answer clearly.

Context:
{context}

Question:
{query}

Answer:
"""

        with st.spinner("Thinking... 🤖"):
            response = llm(prompt, max_length=200)

        st.subheader("🤖 AI Answer")
        st.write(response[0]["generated_text"])

        st.subheader("📚 Sources")
        for r in results:
            st.write("•", r)