from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

embedding = OllamaEmbeddings(
    model="nomic-embed-text"
)

docs = ["Hello world", "LangChain embeddings example"]

vectors = embedding.embed_documents(docs)

print(vectors)