from langgraph.graph import StateGraph, END
from graph.state import AgentState

from agents.supervisor_agent import route_query
from agents.emergency_agent import emergency_agent
from agents.appointment_agent import appointment_agent
from agents.medical_history_agent import medical_history_agent
from agents.general_query_agent import general_query_agent


def build_graph():
    """
    Builds the multi-agent LangGraph for the Medical AI Assistant
    """
    workflow = StateGraph(AgentState)

    # Register all nodes (agents)
    workflow.add_node("supervisor", route_query)
    workflow.add_node("emergency", emergency_agent)
    workflow.add_node("appointment", appointment_agent)
    workflow.add_node("medical_history", medical_history_agent)
    workflow.add_node("general", general_query_agent)

    # Entry point - every conversation starts at the supervisor
    workflow.set_entry_point("supervisor")

    # Conditional routing - supervisor decides which agent runs next
    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state["next_agent"],
        {
            "emergency": "emergency",
            "appointment": "appointment",
            "medical_history": "medical_history",
            "general": "general",
        }
    )

    # After any agent responds, the conversation ends (for this turn)
    workflow.add_edge("emergency", END)
    workflow.add_edge("appointment", END)
    workflow.add_edge("medical_history", END)
    workflow.add_edge("general", END)

    # Compile the graph into a runnable app
    app = workflow.compile()
    return app


if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    app = build_graph()

    # Test 1: General query
    print("=== TEST 1: General Query ===")
    result = app.invoke({
        "messages": [HumanMessage(content="I have a mild headache, what should I do?")],
        "patient_id": None,
        "intent": None,
        "next_agent": None,
        "is_emergency": False
    })
    print(result["messages"][-1].content)
    print()

    # Test 2: Appointment booking
    print("=== TEST 2: Appointment ===")
    result = app.invoke({
        "messages": [HumanMessage(content="I need an appointment with a general physician on 2026-09-26 at 11:00")],
        "patient_id": 1,
        "intent": None,
        "next_agent": None,
        "is_emergency": False
    })
    print(result["messages"][-1].content)
    print()

    # Test 3: Emergency
    print("=== TEST 3: Emergency ===")
    result = app.invoke({
        "messages": [HumanMessage(content="I can't breathe properly and my chest hurts badly")],
        "patient_id": None,
        "intent": None,
        "next_agent": None,
        "is_emergency": False
    })
    print(result["messages"][-1].content)
    print()

    # Test 4: Medical history
    print("=== TEST 4: Medical History ===")
    result = app.invoke({
        "messages": [HumanMessage(content="Show me my past appointments")],
        "patient_id": 1,
        "intent": None,
        "next_agent": None,
        "is_emergency": False
    })
    print(result["messages"][-1].content)