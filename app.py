import os
import PyPDF2
import pinecone
import streamlit as st
import textract
from dotenv import dotenv_values
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.document_loaders.pdf import PyPDFLoader



env_variables = dotenv_values("key.env")

PINECONE_API_KEY = env_variables.get('PINECONE_API_KEY')
PINECONE_ENV = env_variables.get('PINECONE_ENV')
OPENAI_API_KEY = env_variables.get('OPENAI_API_KEY')
embedding_model = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY

TARGET_DIR = os.path.join(os.getcwd(),'data')

def upload_and_save_pdf(uploaded_file):
    if uploaded_file is not None:
        try:
            filename = uploaded_file.name
            filepath = os.path.join(TARGET_DIR, filename)
            with open(filepath, "wb") as f:
                f.write(uploaded_file.read())
            st.success(f"File '{filename}' uploaded and saved successfully!")
            return filename
        except Exception as e:
            st.error(f"An error occurred: {e}")
            return None

def delete_uploaded_file(filename):
    if filename is not None:
        filepath = os.path.join(TARGET_DIR, filename)
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                st.success(f"File '{filename}' deleted successfully!")
            else:
                st.info(f"File '{filename}' not found.")
        except Exception as e:
            st.error(f"An error occurred: {e}")


def doc_preprocessing(filename):
    loader = PyPDFLoader(os.path.join(TARGET_DIR,filename))
    docs = loader.load()
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=0
    )
    docs_split = text_splitter.split_documents(docs)
    return docs_split


def embedding_db(docs_split):
    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
    index_name = 'langchain-demo'
    embeddings = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)
    doc_db = PineconeVectorStore.from_documents(
        docs_split, 
        embeddings, 
        index_name='langchain-demo'
    )
    return doc_db

llm = ChatOpenAI()

def retrival_answer(query, doc_db):
    docs = doc_db.similarity_search(query)
    return docs[0].page_content

def main():
    st.title("Question and Answering App ")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    filename = upload_and_save_pdf(uploaded_file)
    if filename is not None:
        st.button("Cut", on_click=lambda: delete_uploaded_file(filename))
    

    if uploaded_file is not None:
        docs_split = doc_preprocessing(filename)
        doc_db = embedding_db(docs_split)
        text_input = st.text_input("Ask Your Query...")
    if st.button("Ask Query"):
        if len(text_input) > 0:
            st.info("Your Query: " + text_input)
            answer = retrival_answer(text_input, doc_db)
            st.success(answer)
        else:
            st.warning("Please upload a PDF file first.")

if __name__ == "__main__":
    main()