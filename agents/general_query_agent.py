import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from dotenv import load_dotenv

from database.vector_store import retrieve_relevant_chunks

load_dotenv()

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model="openai/gpt-oss-120b",
    temperature=0.3
)

GENERAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a friendly medical AI assistant helping patients in a hospital.

IMPORTANT: Always respond in English only, regardless of the language the user writes in.

You have access to the hospital's internal knowledge base below. Use it whenever it is relevant 
to answer the patient's question accurately. If the knowledge base doesn't contain relevant 
information, answer using your general medical knowledge instead.

Hospital Knowledge Base (relevant excerpts):
{context}

Rules:
- Prioritize information from the hospital knowledge base when relevant
- Provide general health information (explain symptoms, basic precautions)
- Never recommend specific medicines or dosages
- Always advise consulting a doctor for serious or persistent symptoms
- Keep answers short and clear
- If the patient wants to book an appointment, guide them to say "I want to book an appointment\""""),
    ("human", "{query}")
])


def general_query_agent(state):
    """
    Answers general health and hospital-related questions using RAG
    (retrieves relevant hospital knowledge, then generates a grounded answer)
    """
    last_message = state["messages"][-1].content

    # Step 1: Retrieve relevant chunks from the hospital knowledge base
    relevant_chunks = retrieve_relevant_chunks(last_message, top_k=3)
    context = "\n\n".join(relevant_chunks) if relevant_chunks else "No specific hospital policy found for this query."

    # Step 2: Generate the answer using the retrieved context
    chain = GENERAL_PROMPT | llm
    result = chain.invoke({"query": last_message, "context": context})

    return {"messages": [AIMessage(content=result.content)]}


if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    test_state = {
        "messages": [HumanMessage(content="Do I need to fast before a blood test?")]
    }
    result = general_query_agent(test_state)
    print(result["messages"][0].content)