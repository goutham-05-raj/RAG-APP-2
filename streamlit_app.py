import streamlit as st
import os
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chat with your PDF",
    page_icon="📄",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #f0f0f0;
}

.main-header {
    text-align: center;
    padding: 2rem 0 1rem;
}

.main-header h1 {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.main-header p {
    color: #9ca3af;
    font-size: 1.05rem;
}

.answer-box {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(167,139,250,0.3);
    border-radius: 16px;
    padding: 1.5rem;
    margin-top: 1rem;
    backdrop-filter: blur(10px);
    line-height: 1.8;
}

.stTextInput > div > div > input {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(167,139,250,0.4) !important;
    border-radius: 12px !important;
    color: #f0f0f0 !important;
    font-size: 1rem !important;
    padding: 0.75rem 1rem !important;
}

.stFileUploader {
    background: rgba(255,255,255,0.05);
    border: 2px dashed rgba(167,139,250,0.4);
    border-radius: 16px;
    padding: 1rem;
}

.status-badge {
    display: inline-block;
    background: linear-gradient(90deg, #7c3aed, #2563eb);
    color: white;
    padding: 0.3rem 1rem;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📄 Chat with your PDF</h1>
    <p>Upload a PDF and ask any question — powered by Groq & LangChain</p>
</div>
""", unsafe_allow_html=True)

# ── Groq API Key ──────────────────────────────────────────────────────────────
groq_api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))

if not groq_api_key:
    st.error("⚠️ GROQ_API_KEY not found. Please add it in Streamlit Cloud → Settings → Secrets.")
    st.stop()

# ── File Upload ───────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("📁 Upload a PDF document", type="pdf")

if uploaded_file:
    st.markdown('<span class="status-badge">✅ PDF Uploaded</span>', unsafe_allow_html=True)

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("🔍 Reading and indexing your PDF..."):
        # Load & split
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_documents(documents)

        # Embeddings (free, runs in cloud)
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vectorstore = FAISS.from_documents(chunks, embedding_model)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    st.success(f"✅ Indexed **{len(chunks)}** chunks from your PDF!")

    # ── Question Input ────────────────────────────────────────────────────────
    st.markdown("### 💬 Ask a Question")
    query = st.text_input("Type your question here...", placeholder="What is this document about?")

    if query:
        with st.spinner("🤖 Thinking..."):
            llm = ChatGroq(
                api_key=groq_api_key,
                model_name="llama3-8b-8192",
                temperature=0.2
            )

            prompt_template = PromptTemplate(
                input_variables=["context", "question"],
                template="""Use the following context to answer the question accurately and concisely.
If you don't know the answer based on the context, say so clearly.

Context:
{context}

Question: {question}

Answer:"""
            )

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                chain_type="stuff",
                chain_type_kwargs={"prompt": prompt_template}
            )

            result = qa_chain.invoke({"query": query})
            answer = result.get("result", "No answer found.")

        st.markdown("### 🧠 Answer")
        st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

    # Cleanup
    os.unlink(tmp_path)

else:
    st.info("👆 Upload a PDF to get started.")