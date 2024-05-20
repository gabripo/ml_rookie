from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

REVIEWS_CHROMA_PATH = "chroma_data"

loader = PyPDFLoader("data/Bodner_NF_2022.pdf")
reviews = loader.load()

# text_splitter = RecursiveCharacterTextSplitter()
# documents = text_splitter.split_documents(reviews)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
documents = text_splitter.split_documents(reviews)

reviews_vector_db = Chroma.from_documents(
    reviews,
    embedding=OllamaEmbeddings(model="llama3"),
    persist_directory=REVIEWS_CHROMA_PATH,
)
