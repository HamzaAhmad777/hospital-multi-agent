from langchain_core.messages import HumanMessage
from graph.build_graph import build_graph


def run_chat():
    print("=" * 50)
    print("🏥 Medical AI Assistant (Multi-Agent System)")
    print("=" * 50)
    print("Type 'exit' or 'quit' to end the conversation.\n")

    app = build_graph()

    # This holds the conversation state across turns
    state = {
        "messages": [],
        "patient_id": None,
        "intent": None,
        "next_agent": None,
        "is_emergency": False
    }

    # Ask for patient ID once at the start (optional - can be skipped)
    patient_id_input = input("Enter your Patient ID (or press Enter to skip): ").strip()
    if patient_id_input.isdigit():
        state["patient_id"] = int(patient_id_input)
        print(f"✅ Logged in as Patient ID: {state['patient_id']}\n")
    else:
        print("⚠️ Continuing without a patient ID (appointment/history features will be limited)\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("\n👋 Thank you for using the Medical AI Assistant. Stay healthy!")
            break

        if not user_input:
            continue

        # Add the new user message to the conversation history
        state["messages"].append(HumanMessage(content=user_input))

        # Run the graph
        result = app.invoke(state)

        # Update state with the new full message history and reset routing fields
        state["messages"] = result["messages"]
        state["intent"] = None
        state["next_agent"] = None
        state["is_emergency"] = False

        # Print the AI's latest response
        ai_response = result["messages"][-1].content
        print(f"\nAssistant: {ai_response}\n")


if __name__ == "__main__":
    run_chat()