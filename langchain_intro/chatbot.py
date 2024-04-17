from langchain_community.llms import Ollama
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser

review_template_str = """Your job is to use patient
reviews to answer questions about their experience at
a hospital. Use the following context to answer questions.
Be as detailed as possible, but don't make up any information
that's not from the context. If you don't know an answer, say
you don't know.

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
chat_model = Ollama(model="llama2:13b")
output_parser = StrOutputParser()

review_chain = review_prompt_template | chat_model | output_parser

# -----
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.schema.runnable import RunnablePassthrough

REVIEWS_CHROMA_PATH = "chroma_data/"

reviews_vector_db = Chroma(
    persist_directory=REVIEWS_CHROMA_PATH,
    embedding_function=OllamaEmbeddings(model="llama2:13b"),
)

reviews_retriever = reviews_vector_db.as_retriever(k=10)

review_chain = (
    {"context": reviews_retriever, "question": RunnablePassthrough()}
    | review_prompt_template
    | chat_model
    | StrOutputParser()
)
