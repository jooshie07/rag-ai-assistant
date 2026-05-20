import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from retriever import retrieve

st.set_page_config(page_title="Pro RAG AI", page_icon="🤖")
st.title("🤖 Pro RAG AI Assistant")

# Load model safely (NO pipeline)
@st.cache_resource
def load_model():
    model_name = "google/flan-t5-base"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return tokenizer, model

tokenizer, model = load_model()

query = st.text_input("Ask something")

if st.button("Search"):

    if not query:
        st.warning("Please enter a question")
    else:

        # retrieve context
        results = retrieve(query)
        context = "\n".join(results)

        prompt = f"""
Context:
{context}

Question:
{query}

Answer in 2–3 lines using only context.
"""

        # 🔥 DIRECT GENERATION (NO PIPELINE)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

        outputs = model.generate(
            **inputs,
            max_new_tokens=120
        )

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # output
        st.subheader("🤖 AI Answer")
        st.write(answer)

        st.subheader("📚 Sources")
        for r in results:
            st.write("•", r)