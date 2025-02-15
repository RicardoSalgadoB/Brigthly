import os
from dotenv import find_dotenv, load_dotenv
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

def get_documents_from_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    document = loader.load()
    return document

def create_db(data):
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorStore = FAISS.from_documents(data, embedding=embedding)
    return vectorStore

def create_chain(vectorStore):
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        api_key=GOOGLE_API_KEY,
        temperature=0.1,
        max_tokens=1000,
        verbose=True
    )
    
    prompt = ChatPromptTemplate.from_template("""
        Make a summary of the class topics contained in the pdf docuement in the {context}. This summary must not be bigger than 50 words.
    """)
    
    chain = create_stuff_documents_chain(llm=model, prompt=prompt)
        
    retriever = vectorStore.as_retriever(search_kwargs={"k":3})
    retrival_chain = create_retrieval_chain(retriever,chain)
    
    return retrival_chain

def main():
    pdf_path = 'path/to/your/pdf_file_to_be_summarized.pdf'
    doc = get_documents_from_pdf(pdf_path)
    vS = create_db(doc)
    chain = create_chain(vS)
    
    response = chain.invoke({
        "input":""
    })
    
    print(response["answer"])

if __name__ == "__main__":
    main()