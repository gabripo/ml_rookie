import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

CHROMA_PATH = "chroma_data"
DATA_PATH = "data"

data = []
for file in os.listdir(DATA_PATH):
    if file.endswith(".pdf"):
        pdf_path = os.path.join(DATA_PATH, file)
        loader = PyPDFLoader(pdf_path)
        data.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
documents = text_splitter.split_documents(data)

reviews_vector_db = Chroma.from_documents(
    data,
    embedding=OllamaEmbeddings(model="llama3"),
    persist_directory=CHROMA_PATH,
)
