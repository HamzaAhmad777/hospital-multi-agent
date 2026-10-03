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

CONFIRM_YES = ["yes", "yeah", "yep", "yup", "confirm", "correct", "sure", "ok", "okay", "go ahead", "book it", "please book"]
CONFIRM_NO = ["no", "nope", "cancel", "don't", "dont", "not now", "never mind", "nevermind", "stop"]


def extract_date_time(text):
    """
    Uses dateparser's search_dates to find a date/time phrase within a full sentence,
    handling relative terms like 'tomorrow', 'monday', 'next week', etc.
    """
    settings = {
        "PREFER_DATES_FROM": "future",
        "RELATIVE_BASE": datetime.now()
    }
    results = search_dates(text, settings=settings, languages=["en"])

    if not results:
        return None, None

    matched_text, parsed = results[-1]

    # Ignore very short matches (like "I") that dateparser sometimes misreads
    # as a date - a genuine date/time phrase is never this short.
    if len(matched_text.strip()) < 4:
        return None, None

    date_str = parsed.strftime("%Y-%m-%d")
    time_str = parsed.strftime("%H:%M")
    return date_str, time_str


def appointment_agent(state):
    """
    Appointment booking agent - finds doctors, checks slot availability,
    and now asks the patient to explicitly confirm (yes/no) before actually
    creating the appointment in the database.
    """
    last_message = state["messages"][-1].content.strip()
    patient_id = state.get("patient_id")
    pending = state.get("pending_appointment")

    # ---------------- STEP 0: A booking is waiting on yes/no ----------------
    if pending:
        reply = last_message.lower()

        if any(word in reply for word in CONFIRM_YES):
            appointment_id = create_appointment(
                patient_id=pending["patient_id"],
                doctor_id=pending["doctor_id"],
                appointment_date=pending["date"],
                appointment_time=pending["time"],
                reason=pending.get("reason", "General checkup")
            )
            response_text = (
                f"✅ Appointment confirmed and booked!\n\n"
                f"Doctor: {pending['doctor_name']} ({pending['specialization']})\n"
                f"Date: {pending['date']}\n"
                f"Time: {pending['time']}\n"
                f"Appointment ID: {appointment_id}"
            )
            return {"messages": [AIMessage(content=response_text)], "pending_appointment": None}

        elif any(word in reply for word in CONFIRM_NO):
            response_text = "No problem, I've cancelled that booking. Let me know if you'd like to try a different time or doctor."
            return {"messages": [AIMessage(content=response_text)], "pending_appointment": None}

        else:
            response_text = (
                f"Just to confirm — should I book you with **{pending['doctor_name']}** "
                f"({pending['specialization']}) on **{pending['date']} at {pending['time']}**? "
                f"Please reply **yes** to confirm or **no** to cancel."
            )
            return {"messages": [AIMessage(content=response_text)], "pending_appointment": pending}

    # ---------------- STEP 1: Extract date/time (no LLM needed) ----------------
    extracted_date, extracted_time = extract_date_time(last_message)

    # ---------------- STEP 2: Extract specialization/reason using the LLM ----------------
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

    # ---------------- STEP 3: We have a date + time -> propose a slot, don't book yet ----------------
    if details.get("date") and details.get("time"):
        doctor = doctors[0]

        if is_slot_available(doctor["doctor_id"], details["date"], details["time"]):
            pending_appointment = {
                "patient_id": patient_id,
                "doctor_id": doctor["doctor_id"],
                "doctor_name": doctor["full_name"],
                "specialization": doctor["specialization"],
                "date": details["date"],
                "time": details["time"],
                "reason": details.get("reason", "General checkup")
            }
            response_text = (
                f"I found an open slot with **Dr. {doctor['full_name']}** ({doctor['specialization']}) "
                f"on **{details['date']} at {details['time']}**.\n\n"
                f"Shall I go ahead and book it? (yes / no)"
            )
            return {"messages": [AIMessage(content=response_text)], "pending_appointment": pending_appointment}

        else:
            next_slot = get_next_available_slot(doctor["doctor_id"], details["date"], details["time"])
            if next_slot:
                pending_appointment = {
                    "patient_id": patient_id,
                    "doctor_id": doctor["doctor_id"],
                    "doctor_name": doctor["full_name"],
                    "specialization": doctor["specialization"],
                    "date": details["date"],
                    "time": next_slot,
                    "reason": details.get("reason", "General checkup")
                }
                response_text = (
                    f"❌ The {details['time']} slot is already booked with Dr. {doctor['full_name']}.\n\n"
                    f"✅ The next available slot is **{next_slot} on {details['date']}**.\n\n"
                    f"Shall I book that instead? (yes / no)"
                )
                return {"messages": [AIMessage(content=response_text)], "pending_appointment": pending_appointment}
            else:
                response_text = (
                    f"❌ All of Dr. {doctor['full_name']}'s slots on {details['date']} are booked. "
                    f"Please provide a different date."
                )
                return {"messages": [AIMessage(content=response_text)]}

    # ---------------- STEP 4: No date/time yet -> ask for it ----------------
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
        "patient_id": 1,
        "pending_appointment": None
    }
    result = appointment_agent(test_state)
    print(result["messages"][0].content)