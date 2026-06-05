from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# Load PDF
loader = PyPDFLoader("temp.pdf")
docs = loader.load()

# Split text
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(docs)

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Store in vector DB
db = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory="chroma_db"
)

retriever = db.as_retriever()

# LLM
os.environ["GROQ_API_KEY"] = "your_groq_api_key_here"
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)

# Ask question
query = input("Ask question: ")

docs = retriever.invoke(query)

context = "\n".join([d.page_content for d in docs])

prompt = f"""
Answer based only on this context:

{context}

Question: {query}
"""

response = llm.invoke(prompt)

print("\nAnswer:\n", response)