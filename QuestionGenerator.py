name = "Chemistry 500"
description = """
This is a cours about chemistry, we explore electronic configuration
"""
text = """
Electron Configuration: Understanding the Distribution of Electrons in Atoms

Electron configuration is a fundamental concept in chemistry and physics that describes the distribution of electrons within the atomic orbitals of an atom or ion. It provides crucial insights into the structure of atoms, influencing the chemical properties, reactivity, and behavior of elements. Understanding electron configurations allows scientists to predict and explain trends across the periodic table, such as atomic size, ionization energy, and electronegativity.
1. The Basics of Electron Configuration

Electrons in an atom are organized into shells, subshells, and orbitals based on principles derived from quantum mechanics. The shells are designated by the principal quantum number nn (1, 2, 3, etc.), which indicates the energy level and relative distance of the electrons from the nucleus. Each shell can contain subshells identified by the letters s, p, d, and f, each with a distinct shape and energy.

The Pauli exclusion principle states that no two electrons can have the same set of four quantum numbers, meaning each orbital can hold a maximum of two electrons with opposite spins. The Aufbau principle dictates that electrons fill orbitals starting with the lowest energy level and move to higher levels as needed. Additionally, Hund’s rule emphasizes that electrons will occupy degenerate orbitals (orbitals with the same energy) singly before pairing up, minimizing electron repulsion.
2. The Notation of Electron Configuration

The electron configuration of an atom is written using a notation that specifies the occupied subshells and the number of electrons in each. For example, the electron configuration of hydrogen is simply 1s11s1, indicating one electron in the 1s orbital. For carbon (atomic number 6), the configuration is written as:
1s22s22p2
1s22s22p2

This notation shows that carbon has two electrons in the 1s orbital, two in the 2s orbital, and two in the 2p orbital.

To simplify the representation for atoms with larger atomic numbers, noble gas notation is often used. For instance, instead of writing out the full configuration for calcium (atomic number 20), which is 1s22s22p63s23p64s21s22s22p63s23p64s2, it can be abbreviated as:
[Ar]4s2
[Ar]4s2

Here, [Ar] represents the electron configuration of the noble gas argon, which precedes calcium in the periodic table.
3. Periodic Trends and Electron Configuration

The arrangement of electrons in an atom is closely related to its position in the periodic table. Elements in the same group (vertical columns) typically have similar outer electron configurations, which accounts for their similar chemical properties. For instance, the alkali metals (group 1) all have a single electron in their outermost s orbital, giving them a strong tendency to lose that electron and form cations.

    Ionization Energy: The energy required to remove an electron from an atom decreases down a group and increases across a period from left to right. This trend can be explained by the increasing nuclear charge across a period, which pulls the electrons closer to the nucleus, making them harder to remove.

    Atomic Radius: The atomic radius decreases across a period due to the increasing effective nuclear charge, which draws electrons closer to the nucleus. However, it increases down a group as additional electron shells are added, increasing the distance between the nucleus and the outermost electrons.

    Electron Affinity and Electronegativity: These properties reflect an atom's ability to attract electrons. Nonmetals, particularly those in the upper right corner of the periodic table, tend to have higher electron affinities and electronegativities because their valence shells are nearly full, and they require only a few more electrons to achieve a stable configuration.

4. Applications of Electron Configuration

Understanding electron configurations is essential for predicting the chemical behavior of elements. For example, the electron configuration of transition metals often involves partially filled d orbitals, which contribute to their ability to form complex ions, exhibit multiple oxidation states, and act as catalysts in chemical reactions.

Electron configuration also plays a key role in determining the magnetic properties of materials. Elements or ions with unpaired electrons exhibit paramagnetism, while those with all paired electrons are diamagnetic. For instance, oxygen (O2O2​) is paramagnetic due to its two unpaired electrons in degenerate molecular orbitals.

In modern technology, electron configurations help in the design of materials with specific electrical, optical, and magnetic properties. For instance, the behavior of semiconductors is influenced by the electron configurations of the elements involved, which in turn affects the performance of devices like solar cells, transistors, and LEDs.
5. Electron Configuration of Ions

When atoms gain or lose electrons to form ions, their electron configurations change. For cations (positively charged ions), electrons are removed starting from the outermost shell. For example, the configuration for sodium (Na) is 1s22s22p63s11s22s22p63s1. When it loses one electron to form Na+Na+, the resulting configuration is 1s22s22p61s22s22p6, the same as the noble gas neon.

For anions (negatively charged ions), electrons are added to the next available orbital. For example, the configuration for chlorine (Cl) is 1s22s22p63s23p51s22s22p63s23p5. When it gains one electron to form Cl−Cl−, the configuration becomes 1s22s22p63s23p61s22s22p63s23p6, equivalent to the noble gas argon.
6. Conclusion

Electron configuration provides a detailed understanding of the electronic structure of atoms, which is fundamental to the study of chemistry and materials science. By understanding how electrons are distributed in atoms and ions, scientists can predict and explain a wide range of physical and chemical properties. From explaining the periodic trends to influencing the design of modern technological materials, the concept of electron configuration remains central to both theoretical studies and practical applications in science.
"""

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
from langchain_core.messages import AIMessage

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

def create_chain():
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        api_key=GOOGLE_API_KEY,
        temperature=0.1,
        max_tokens=1000,
        verbose=True
    )
    
    prompt = ChatPromptTemplate.from_template("""
        Generate 10 multiple-answer questions based on the description of this topic: {description}, its name: {name} and its content: {text}.\
        Include the multple answer options. And then include the correct answer.\
        Then create a prompt so that an image related to the question can be generated.\
        Finally, parse your output using the {format_instructions}."),
    """)
    
    parser = JsonOutputParser(pydantic_object=DesiredOutput)
    
    chain = prompt | model | parser
    
    return chain, parser

                                                            
def generate_10_questions(name, description, text):
    num_questions = 10

    chain, parser = create_chain()
    
    output = chain.invoke({
        "name": name,
        "description": description,
        "text": text,
        "format_instructions": parser.get_format_instructions()
    })

if __name__ == '__main__':
    generate_10_questions()