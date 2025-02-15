import time

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

def find_lag():
    return 5

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
        temperature=0,
        max_tokens=1000,
        verbose=True
    )
    
    prompt = ChatPromptTemplate.from_template(
    """
    {student_name} is a university student feeling {emotions} with the contents of the topic {topic}. 
    The student is {lag} topics behind the rest of the class in this course.
    Recommend a method (including meditation and mindfulness) in which he can become better academically and/or emotionally.
    Use the academic contents of the topic included in {context}.
    
    Include only 3 recommendations.
    Answer as if speaking directly with Diego in a paternalistic manner.
    """
    )
    
    chain = create_stuff_documents_chain(llm=model, prompt=prompt)
    
    retriever = vectorStore.as_retriever(search_kwargs={"k":3})
    retrival_chain = create_retrieval_chain(retriever,chain)
    
    return retrival_chain
    
def main():
    emotions = "happy, excited"
    topic = "Differences between Guerilla Movements in Mexico and Syria"
    
    emotions_list = emotions.split(", ")
    emotions = ' and '.join(emotions_list)
        
    #data_csv = 'path/to/your/data.csv'
    
    topic_content_path = '/Users/r.salgadob./Brillantemente/Data/Bibliografía anotada 2.pdf'
    lag = find_lag()
    
    doc = get_documents_from_pdf(topic_content_path)
    vS = create_db(doc)
    chain = create_chain(vS)
    
    name = "Fernando" #input("Student's name: ")
    
    output = chain.stream({
        "student_name": name,
        "emotions": emotions,
        "topic": topic,
        "lag": lag,
        "input": ""
    })
    
    for s in output:
        d = dict(s)
        print(list(d.values())[0], end="")
    
if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(end-start)