import os
import json
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from dateparser.search import search_dates
from dotenv import load_dotenv

from database.models import (
    get_all_doctors,
    get_doctors_by_specialization,
    create_appointment,
    is_slot_available,
    get_next_available_slot,
    get_patient_by_id
)

load_dotenv()

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model="openai/gpt-oss-120b",
    temperature=0
)

EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a medical appointment assistant. Extract the following details from 
the user's message and return ONLY valid JSON, nothing else:

{{"specialization": "<doctor's specialization if mentioned, else null>", "reason": "<short reason for the appointment>"}}

Do not include date or time - those are handled separately. Return ONLY the JSON object."""),
    ("human", "{query}")
])


def extract_date_time(text):
    """
    Uses dateparser's search_dates to find a date/time phrase within a full sentence,
    handling relative terms like 'tomorrow', 'monday', 'next week', etc.
    """
    settings = {
        "PREFER_DATES_FROM": "future",
        "RELATIVE_BASE": datetime.now()
    }
    results = search_dates(text, settings=settings)

    if not results:
        return None, None

    # search_dates returns a list of (matched_text, datetime) tuples - take the last match
    _, parsed = results[-1]

    date_str = parsed.strftime("%Y-%m-%d")
    time_str = parsed.strftime("%H:%M")
    return date_str, time_str

def appointment_agent(state):
    """
    Appointment booking agent - finds doctors, checks slot availability,
    and creates the appointment (with double-booking prevention)
    """
    last_message = state["messages"][-1].content
    patient_id = state.get("patient_id")

    # Step 1: Extract date/time using dateparser (reliable, no LLM needed)
    extracted_date, extracted_time = extract_date_time(last_message)

    # Step 2: Extract specialization/reason using the LLM
    chain = EXTRACTION_PROMPT | llm
    result = chain.invoke({"query": last_message})

    raw_output = result.content.strip()
    if raw_output.startswith("```"):
        raw_output = raw_output.strip("`").replace("json", "", 1).strip()

    try:
        llm_details = json.loads(raw_output)
    except json.JSONDecodeError:
        llm_details = {"specialization": None, "reason": last_message}

    extracted_reason = llm_details.get("reason")
    if not extracted_reason or extracted_reason.strip().lower() == "none":
        extracted_reason = "General checkup"

    details = {
        "specialization": llm_details.get("specialization"),
        "date": extracted_date,
        "time": extracted_time,
        "reason": extracted_reason
}

    if not patient_id:
        response_text = (
            "To book an appointment, I first need your patient ID. "
            "If you're a new patient, please provide your name, date of birth, and phone number."
        )
        return {"messages": [AIMessage(content=response_text)]}

    if details.get("specialization"):
        doctors = get_doctors_by_specialization(details["specialization"])
    else:
        doctors = get_all_doctors()

    if not doctors:
        response_text = "Sorry, no doctors are available for this specialization right now."
        return {"messages": [AIMessage(content=response_text)]}

    if details.get("date") and details.get("time"):
        doctor = doctors[0]

        if is_slot_available(doctor["doctor_id"], details["date"], details["time"]):
            appointment_id = create_appointment(
                patient_id=patient_id,
                doctor_id=doctor["doctor_id"],
                appointment_date=details["date"],
                appointment_time=details["time"],
                reason=details.get("reason", "General checkup")
            )
            response_text = (
                f"✅ Appointment booked successfully!\n\n"
                f"Doctor: {doctor['full_name']} ({doctor['specialization']})\n"
                f"Date: {details['date']}\n"
                f"Time: {details['time']}\n"
                f"Appointment ID: {appointment_id}"
            )
        else:
            next_slot = get_next_available_slot(doctor["doctor_id"], details["date"], details["time"])
            if next_slot:
                response_text = (
                    f"❌ The {details['time']} slot is already booked with Dr. {doctor['full_name']}.\n\n"
                    f"✅ Next available slot: {next_slot} on {details['date']}\n\n"
                    f"Would you like to confirm this time?"
                )
            else:
                response_text = (
                    f"❌ All of Dr. {doctor['full_name']}'s slots on {details['date']} are booked. "
                    f"Please provide a different date."
                )
    else:
        doctor_list = "\n".join(
            [f"- Dr. {d['full_name']} ({d['specialization']})" for d in doctors]
        )
        response_text = (
            f"Here are the available doctors:\n{doctor_list}\n\n"
            f"Please let me know your preferred date and time."
        )

    return {"messages": [AIMessage(content=response_text)]}


if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    test_state = {
        "messages": [HumanMessage(content="I need an appointment with a general physician tomorrow at 3pm")],
        "patient_id": 1
    }
    result = appointment_agent(test_state)
    print(result["messages"][0].content)