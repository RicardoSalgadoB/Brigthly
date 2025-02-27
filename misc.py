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
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

def get_documents_from_csv(csv_path):
    loader = CSVLoader(file_path=csv_path, encoding="utf-8", csv_args={'delimiter':','})
    data = loader.load()
    return data

def create_db(data):
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorStore = FAISS.from_documents(data, embedding=embedding)
    return vectorStore

class JSON_Object(BaseModel):
    x: str = Field(description="Joe Biden")

def create_chain(vectorStore):
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        api_key=GOOGLE_API_KEY,
        temperature=0.7,
        max_tokens=1000,
        verbose=True
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's question based on the context: {context}. \nFormatting Instructions: {format_instructions}"),
        ("human", "{input}")
    ])
    
    parser = JsonOutputParser(pydantic_object=JSON_Object)
        
    chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt
    )
    
    chain = chain | parser
    
    retriever = vectorStore.as_retriever(search_kwargs={"k":3})
    
    retrival_chain = create_retrieval_chain(
        retriever,
        chain
    )
    
    return retrival_chain, parser

if __name__ == '__main__':
    csv_path = '/Users/r.salgadob./Brillantemente/Data/profesores_d.csv'
    data = get_documents_from_csv(csv_path)
    vS = create_db(data)
    chain, parser = create_chain(vS)
    
    user_inpt = input("You: ")
    
    response = chain.invoke({
        "input": user_inpt,
        "format_instructions": parser.get_format_instructions()
    })
    
    print(response)
    
    
#output = [question, answer, prompt]    