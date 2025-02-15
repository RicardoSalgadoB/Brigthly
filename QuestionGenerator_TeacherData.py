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
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

class DesiredOutput(BaseModel):
    question1: str = Field(description="The 1st question, generated from the context")
    answer1: str = Field(description="The multiple possible answers to the 1st question")
    correct_answer1: str = Field(description="This is the correct answer to the 1st question")
    prompt1: str = Field(description="A prompt to generate an image based o n the 1st question")
    
    question2: str = Field(description="The 2nd question, generated from the context")
    answer2: str = Field(description="The multiple possible answers to the 2nd question")
    correct_answer2: str = Field(description="This is the correct answer to the end question")
    prompt2: str = Field(description="A prompt to generate an image based o n the 2nd question")
    
    question3: str = Field(description="The 3rd question, generated from the context")
    answer3: str = Field(description="The multiple possible answers to the 3rd question")
    correct_answer3: str = Field(description="This is the correct answer to the 3rd question")
    prompt3: str = Field(description="A prompt to generate an image based o n the 3rd question")
    
    question4: str = Field(description="The 4th question, generated from the context")
    answer4: str = Field(description="The multiple possible answers to the 4th question")
    correct_answer4: str = Field(description="This is the correct answer to the 4th question")
    prompt4: str = Field(description="A prompt to generate an image based o n the 4th question")
    
    question5: str = Field(description="The 5th question, generated from the context")
    answer5: str = Field(description="The multiple possible answers to the previous 5th question")
    correct_answer5: str = Field(description="This is the correct answer to the 5th question")
    prompt5: str = Field(description="A prompt to generate an image based o n the 5th question")
    
    question6: str = Field(description="The 6th question, generated from the context")
    answer6: str = Field(description="The multiple possible answers to the 6th question")
    correct_answer6: str = Field(description="This is the correct answer to the 6th question")
    prompt6: str = Field(description="A prompt to generate an image based o n the 6th question")
    
    question7: str = Field(description="The 7th question, generated from the context")
    answer7: str = Field(description="The multiple possible answers to the 7th question")
    correct_answer7: str = Field(description="This is the correct answer to the 7th question")
    prompt7: str = Field(description="A prompt to generate an image based o n the 7th question")
    
    question8: str = Field(description="The 8th question, generated from the context")
    answer8: str = Field(description="The multiple possible answers to the 8th question")
    correct_answer8: str = Field(description="This is the correct answer to the 8th question")
    prompt8: str = Field(description="A prompt to generate an image based o n the 8th question")
    
    question9: str = Field(description="The 9th question, generated from the context")
    answer9: str = Field(description="The multiple possible answers to the 9th question")
    correct_answer9: str = Field(description="This is the correct answer to the 9th question")
    prompt9: str = Field(description="A prompt to generate an image based o n the 9th question")
    
    question10: str = Field(description="The 10th question, generated from the context")
    answer10: str = Field(description="The multiple possible answers to the 10th question")
    correct_answer10: str = Field(description="This is the correct answer to the 10th question")
    prompt10: str = Field(description="A prompt to generate an image based o n the 10th question")

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
        max_tokens=2000,
        verbose=True
    )
    
    prompt = ChatPromptTemplate.from_template("""
        Generate 10 multiple-answer questions based on the following contents of the following document: {context}.\
        Include the multple answer options. And then include the correct answer.\
        Then create a prompt so that an image related to the question can be generated by a Gen AI Image generator.\
        Finally, parse your output using the {format_instructions}.
    """)
    
    parser = JsonOutputParser(pydantic_object=DesiredOutput)
    
    chain = create_stuff_documents_chain(llm=model, prompt=prompt, output_parser=parser)
        
    retriever = vectorStore.as_retriever(search_kwargs={"k":3})
    retrival_chain = create_retrieval_chain(retriever,chain)
    
    return retrival_chain, parser

def main():
    pdf_path = 'path/to/your/pdf.pdf'
    doc = get_documents_from_pdf(pdf_path)
    vS = create_db(doc)
    chain, parser = create_chain(vS)
    
    response = chain.invoke({
        "input": "",
        "format_instructions": parser.get_format_instructions(),
    })
    
    print(response)
    
    
if __name__ == '__main__':
    main()