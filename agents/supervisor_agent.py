import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Groq LLM setup
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model="openai/gpt-oss-120b",
    temperature=0
)

# Supervisor prompt - decides which agent should handle the query
SUPERVISOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are the supervisor of a medical AI system. Your job is to read the 
patient's message and decide which category it belongs to.

Categories:
- "emergency": If the patient's life is at risk (chest pain, severe bleeding, unconsciousness, 
  breathing difficulty, severe accident, etc.)
- "appointment": If the patient wants to book/cancel/check an appointment, or find a doctor
- "medical_history": If the patient wants to see their past history, previous appointments, or records
- "general": Any other general health questions or normal conversation

Return only one word: emergency, appointment, medical_history, or general
Do not write anything else, just the category name."""),
    ("human", "{query}")
])


def route_query(state):
    """
    Supervisor node - looks at the user's last message and decides
    which agent should run next
    """
    # If an appointment is waiting on a yes/no confirmation, keep routing to the
    # appointment agent regardless of what the LLM would classify this reply as.
    # Otherwise a short reply like "yes" could get misclassified as "general".
    if state.get("pending_appointment"):
        return {
            "intent": "appointment",
            "next_agent": "appointment",
            "is_emergency": False
        }

    last_message = state["messages"][-1].content

    chain = SUPERVISOR_PROMPT | llm
    result = chain.invoke({"query": last_message})

    intent = result.content.strip().lower()

    # Safety check - if the LLM returns something unexpected, default to "general"
    valid_intents = ["emergency", "appointment", "medical_history", "general"]
    if intent not in valid_intents:
        intent = "general"

    is_emergency = (intent == "emergency")

    return {
        "intent": intent,
        "next_agent": intent,
        "is_emergency": is_emergency
    }


if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    test_state = {
        "messages": [HumanMessage(content="I need an appointment with a doctor tomorrow")],
        "pending_appointment": None
    }
    result = route_query(test_state)
    print("Result:", result)