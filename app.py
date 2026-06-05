from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings# 1 Load PDF
loader = PyPDFLoader("Love-Stories.pdf")
documents = loader.load()

# 2 Split text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

docs = text_splitter.split_documents(documents)

# 3 Embeddings
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4 Vector DB (FAST VERSION)
if os.path.exists("vector_db"):
    print("Loading existing vector database...")
    vectorstore = FAISS.load_local("vector_db", embedding, allow_dangerous_deserialization=True)
else:
    print("Creating new vector database...")
    vectorstore = FAISS.from_documents(docs, embedding)
    vectorstore.save_local("vector_db")

# 5 LLM
os.environ["GROQ_API_KEY"] = "your_groq_api_key_here"
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)

# 6 Query
query = input("Ask a question from the PDF: ")

retriever = vectorstore.as_retriever()
relevant_docs = retriever.invoke(query)

context = "\n".join([doc.page_content for doc in relevant_docs])

prompt = f"""
Use the context to answer the question.

Context:
{context}

Question:
{query}
"""

response = llm.invoke(prompt)

print("\nAnswer:\n")
print(response)