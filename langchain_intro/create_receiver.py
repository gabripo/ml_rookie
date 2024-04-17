from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

REVIEWS_CSV_PATH = "data/reviews.csv"
REVIEWS_CHROMA_PATH = "chroma_data"

loader = CSVLoader(file_path=REVIEWS_CSV_PATH, source_column="review")
reviews = loader.load()

text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(reviews)

reviews_vector_db = Chroma.from_documents(
    documents,
    embedding=OllamaEmbeddings(model="llama2:13b"),
    persist_directory=REVIEWS_CHROMA_PATH,
)
