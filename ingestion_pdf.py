import os
from queue import PriorityQueue

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader #change 1
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
load_dotenv()

if __name__ == "__main__":
    print("Ingesting...")
    loader = PyPDFLoader("INTRODUCTIONTOHISTORY-PART-1.pdf") #change 2
    documents = loader.load()

    print("splitting...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    print(f"created {len(texts)} chunks")

    print("embedding and ingesting...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    PineconeVectorStore.from_documents(
        texts,
        embeddings,
        index_name=os.environ["INDEX_NAME_PDF"],
    )
    print("finish")