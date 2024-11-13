import os
from dotenv import find_dotenv, load_dotenv
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.output_parsers import JsonOutputParser

def get_documents_from_csv(csv_path):
    loader = CSVLoader(file_path=csv_path)
    data = loader.load()
    return data

def create_db(data):
    embedding = GoogleGenerativeAIEmbeddings()
    vectorStore = FAISS.from_documents(data, embedding=embedding)
    return vectorStore

def create_chain(data):
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        api_key=GOOGLE_API_KEY,
        temperature=0.7,
        max_tokens=1000,
        verbose=True
    )
    
    prompt = ChatPromptTemplate.from_messages([
            ("system", "Write a summary of the positive comments include in the following context: {context}"),
            ()
    ])

if __main__ == '__name__':
    data = get_documents_from_csv()