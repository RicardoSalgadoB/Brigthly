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

class DesiredOutput(BaseModel):
    question: str = Field(description="The question, generated from the context")
    answer: str = Field(description="The answer to the previous question")
    prompt: str = Field(description="A prompt to generate an image based o n the question")

def create_chain(vectorStore):
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        api_key=GOOGLE_API_KEY,
        temperature=0.7,
        max_tokens=1000,
        verbose=True
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Generate a question. Use the name of the teacher as the answer and use the characteristics\
            described in the comments in the question to help the user guess the answer. Also generate a prompt\
                to generate a caricature of the teacher based on the comments using a Gen AI model: {context}.\
                    Then parse your output using the {format_instructions}."),
        ("human", "{input}")
    ])
    
    parser = JsonOutputParser(pydantic_object=DesiredOutput)
    
    chain = create_stuff_documents_chain(llm=model, prompt=prompt, output_parser=parser)
        
    retriever = vectorStore.as_retriever(search_kwargs={"k":3})
    retrival_chain = create_retrieval_chain(retriever,chain)
    
    return retrival_chain, parser

def main():
    csv_path = '/Users/r.salgadob./Brillantemente/Data/profesores_d.csv'
    data = get_documents_from_csv(csv_path)
    vS = create_db(data)
    chain, parser = create_chain(vS)
    
    user_inpt = input("You: ")
    
    response = chain.invoke({
        "input": user_inpt,
        "format_instructions": parser.get_format_instructions()
    })
    
    print(response["answer"])
    question = response["answer"]["question"]
    answer = response["answer"]["answer"]
    prompt = response["answer"]["prompt"]
    
    output = [question, answer, prompt]
    
    
if __name__ == '__main__':
    main()