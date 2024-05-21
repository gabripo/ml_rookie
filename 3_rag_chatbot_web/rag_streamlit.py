import os
from langchain_community.vectorstores import Chroma
import chromadb
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

import streamlit as st

CHROMA_PATH = "chroma_data"
DATA_PATH = "data"

review_template_str = """You are an expert of Nuclear Physics. Reply to questions concerning Nuclear Fusion.

{context}
"""

review_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["context"],
        template=review_template_str,
    )
)
review_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["question"],
        template="{question}",
    )
)
messages = [review_system_prompt, review_human_prompt]

review_prompt_template = ChatPromptTemplate(
    input_variables=["context", "question"],
    messages=messages,
)
chat_model = Ollama(model="llama3")


def chroma_db_persist_pdf(
    pdf_folder_path=DATA_PATH, chroma_store_path=CHROMA_PATH
) -> Chroma:
    data = []
    for file in os.listdir(pdf_folder_path):
        if file.endswith(".pdf"):
            single_pdf_path = os.path.join(pdf_folder_path, file)
            loader = PyPDFLoader(single_pdf_path)
            data.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    split_documents = text_splitter.split_documents(data)

    client = chromadb.Client()
    if client.list_collections():
        print("Creating new collection")
        consent_collection = client.create_collection("consent_collection")
    else:
        print("Collection already exists")
    vectordb = Chroma.from_documents(
        documents=split_documents,
        embedding=OllamaEmbeddings(model="llama3"),
        persist_directory=chroma_store_path,
    )
    vectordb.persist()
    return vectordb


def get_llm_response(query):
    if os.path.exists(CHROMA_PATH):
        print("Vector database already existing!")
        vectordb = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=OllamaEmbeddings(model="llama3"),
        )
    else:
        vectordb = chroma_db_persist_pdf()
    print("Vector database loaded")
    num_reviews_to_consider = 20
    reviews_retriever = vectordb.as_retriever(k=num_reviews_to_consider)
    review_chain = (
        {"context": reviews_retriever, "question": RunnablePassthrough()}
        | review_prompt_template
        | chat_model
        | StrOutputParser()
    )
    print("Passing the query...")
    answer = review_chain.invoke(query)
    print("Query replied!")
    return answer


# Streamlit UI
# ===============
st.set_page_config(page_title="Doc Searcher", page_icon=":robot:")
st.header("Query PDF Source")

form_input = st.text_input("Enter Query")
submit = st.button("Generate")

if submit:
    st.write(get_llm_response(form_input))
