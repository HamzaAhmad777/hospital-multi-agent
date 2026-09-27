from langchain_core.messages import AIMessage


def emergency_agent(state):
    """
    When an emergency is detected, this agent immediately responds
    and directs the patient toward emergency services
    """
    response_text = (
        "🚨 This looks like an EMERGENCY! 🚨\n\n"
        "Please immediately:\n"
        "1. Go to the nearest emergency room, or\n"
        "2. Call an ambulance at 1122 (Pakistan)\n"
        "3. Get help from someone nearby\n\n"
        "I am an AI assistant and cannot provide medical advice in emergency situations. "
        "Please seek professional medical help immediately."
    )

    return {
        "messages": [AIMessage(content=response_text)]
    }


if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    test_state = {
        "messages": [HumanMessage(content="I am having chest pain")]
    }
    result = emergency_agent(test_state)
    print(result["messages"][0].content)