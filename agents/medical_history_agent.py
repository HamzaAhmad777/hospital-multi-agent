from langchain_core.messages import AIMessage
from database.models import get_appointments_by_patient, get_patient_by_id


def medical_history_agent(state):
    """
    Shows the patient's medical history (past appointments)
    """
    patient_id = state.get("patient_id")

    if not patient_id:
        response_text = (
            "To view your history, I first need your patient ID. "
            "Please provide your name or patient ID."
        )
        return {"messages": [AIMessage(content=response_text)]}

    patient = get_patient_by_id(patient_id)
    if not patient:
        response_text = "Sorry, this patient record was not found."
        return {"messages": [AIMessage(content=response_text)]}

    appointments = get_appointments_by_patient(patient_id)

    if not appointments:
        response_text = f"{patient['full_name']} has no past appointment records."
        return {"messages": [AIMessage(content=response_text)]}

    history_text = f"📋 Medical History for {patient['full_name']}:\n\n"
    for appt in appointments:
        history_text += (
            f"• Date: {appt['appointment_date']} | Time: {appt['appointment_time']}\n"
            f"  Doctor: {appt['doctor_name']} ({appt['specialization']})\n"
            f"  Reason: {appt['reason']}\n"
            f"  Status: {appt['status']}\n\n"
        )

    return {"messages": [AIMessage(content=history_text)]}


if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    test_state = {
        "messages": [HumanMessage(content="Show me my past appointments")],
        "patient_id": 1
    }
    result = medical_history_agent(test_state)
    print(result["messages"][0].content)