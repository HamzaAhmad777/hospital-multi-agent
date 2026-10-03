import streamlit as st
import base64
from datetime import date, datetime
from langchain_core.messages import HumanMessage, AIMessage
from graph.build_graph import build_graph
from database.models import (
    get_patient_by_id,
    get_patient_by_phone,
    phone_already_registered,
    create_patient,
    get_all_doctors,
    get_doctors_by_specialization,
    create_appointment,
    get_appointments_by_patient,
    is_slot_available,
    get_next_available_slot,
)

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="MediCare AI | Medical Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ BACKGROUND + FONTS + THEME ------------------
def set_background(image_path):
    try:
        with open(image_path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        bg_css = f"""
            background: linear-gradient(180deg, rgba(6, 20, 45, 0.82) 0%, rgba(6, 20, 45, 0.92) 100%),
                        url(data:image/jpg;base64,{data});
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        """
    except FileNotFoundError:
        bg_css = """
            background: radial-gradient(circle at 20% 10%, #0f2a4a 0%, #061426 55%, #030a16 100%);
        """

    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

        html, body, .stMarkdown, .stChatMessage, .stTextInput, p, span, div[data-testid="stMarkdownContainer"] {{
            font-family: 'Inter', sans-serif;
            color: #e6ecf5;
        }}

        .stApp {{
            {bg_css}
        }}

        #MainMenu, footer {{visibility: hidden;}}

        .app-header {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 14px;
            padding: 20px 0 6px 0;
        }}
        .app-header .icon-box {{
            width: 54px;
            height: 54px;
            border-radius: 14px;
            background: linear-gradient(135deg, #2563eb, #0ea5e9);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 26px;
            box-shadow: 0 6px 18px rgba(37, 99, 235, 0.45);
        }}
        .app-title {{
            font-family: 'Poppins', sans-serif;
            font-size: 2.1rem;
            font-weight: 700;
            color: #f4f7fb;
            margin: 0;
            letter-spacing: -0.5px;
        }}
        .app-subtitle {{
            text-align: center;
            color: #93a5c4;
            font-size: 0.95rem;
            font-weight: 500;
            margin-bottom: 20px;
        }}

        .stChatMessage {{
            background: rgba(255, 255, 255, 0.045) !important;
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px !important;
            padding: 14px 16px !important;
            backdrop-filter: blur(10px);
            margin-bottom: 8px;
        }}
        .stChatMessage p {{
            color: #e6ecf5 !important;
            font-size: 0.96rem;
            line-height: 1.55;
        }}

        [data-testid="stChatInput"] textarea {{
            color: #1a1a1a !important;
            background-color: #ffffff !important;
        }}

        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #08152b 0%, #050d1c 100%);
            border-right: 1px solid rgba(255,255,255,0.06);
        }}
        section[data-testid="stSidebar"] * {{
            color: #dbe4f3 !important;
            font-family: 'Inter', sans-serif;
        }}
        .sidebar-card {{
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 14px 16px;
            margin-bottom: 16px;
        }}
        .status-dot {{
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 6px;
        }}
        .status-online {{ background: #22c55e; box-shadow: 0 0 8px #22c55e; }}

        section[data-testid="stSidebar"] input {{
            background: #ffffff !important;
            border-radius: 8px !important;
            border: 1px solid rgba(255,255,255,0.15) !important;
            color: #1a1a1a !important;
        }}
        section[data-testid="stSidebar"] input::placeholder {{
            color: #6b7280 !important;
        }}

        div[data-testid="stForm"] input, div[data-testid="stForm"] textarea {{
            background: #ffffff !important;
            color: #1a1a1a !important;
        }}

        .metric-card {{
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 14px;
            padding: 18px;
            text-align: center;
        }}
        .metric-number {{
            font-size: 2rem;
            font-weight: 700;
            color: #38bdf8;
        }}
        .metric-label {{
            color: #93a5c4;
            font-size: 0.85rem;
        }}
        .emergency-banner {{
            background: rgba(239, 68, 68, 0.15);
            border: 1px solid rgba(239, 68, 68, 0.5);
            border-radius: 12px;
            padding: 14px 18px;
            margin-bottom: 16px;
            color: #fecaca;
        }}
    </style>
    """, unsafe_allow_html=True)

set_background("assets/background.jpg")

# ------------------ SESSION STATE DEFAULTS ------------------
defaults = {
    "patient_id": None,
    "patient_name": None,
    "view": "💬 Ask Assistant",
    "messages": [],
    "graph_app": None,
    "pending_appointment": None,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ==================================================================
# LOGIN GATE — nothing else renders until the patient logs in
# ==================================================================
if not st.session_state["patient_id"]:
    st.markdown("""
    <div class="app-header">
        <div class="icon-box">🏥</div>
        <div class="app-title">MediCare AI Assistant</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">Your intelligent multi-agent healthcare companion</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.4, 1])
    with col2:
        st.markdown("### Welcome — are you a new or existing patient?")
        tab1, tab2 = st.tabs(["🔑 Existing Patient", "📝 New Patient"])

        with tab1:
            login_phone = st.text_input("Phone Number", placeholder="03001234567", key="login_phone")
            if st.button("Login", use_container_width=True, key="login_btn"):
                if login_phone.strip():
                    patient = get_patient_by_phone(login_phone.strip())
                    if patient:
                        st.session_state["patient_id"] = patient["patient_id"]
                        st.session_state["patient_name"] = patient["full_name"]
                        st.rerun()
                    else:
                        st.error("No account found with this phone number. Please register as a new patient.")
                else:
                    st.warning("Please enter your phone number.")

        with tab2:
            reg_name = st.text_input("Full Name", key="reg_name")
            reg_gender = st.selectbox("Gender", ["Male", "Female", "Other"], key="reg_gender")
            reg_dob = st.date_input(
                "Date of Birth",
                key="reg_dob",
                min_value=date(1920, 1, 1),
                max_value=date.today()
                )
            reg_phone = st.text_input("Phone Number", placeholder="03001234567", key="reg_phone")
            reg_address = st.text_area("Address", key="reg_address", height=68)

            if st.button("Register", use_container_width=True, key="register_btn"):
                if not reg_name.strip() or not reg_phone.strip():
                    st.warning("Name and phone number are required.")
                elif phone_already_registered(reg_phone.strip()):
                    st.error("This phone number is already registered. Please use Login instead.")
                else:
                    new_id = create_patient(
                        full_name=reg_name.strip(),
                        gender=reg_gender,
                        date_of_birth=reg_dob,
                        phone=reg_phone.strip(),
                        address=reg_address.strip()
                    )
                    st.session_state["patient_id"] = new_id
                    st.session_state["patient_name"] = reg_name.strip()
                    st.rerun()

    st.stop()  # everything below is only reached once logged in

# ==================================================================
# LOGGED-IN AREA
# ==================================================================
if st.session_state["graph_app"] is None:
    st.session_state["graph_app"] = build_graph()

# ------------------ SIDEBAR NAVIGATION ------------------
with st.sidebar:
    st.markdown(
        f'<div class="sidebar-card"><span class="status-dot status-online"></span>'
        f'Welcome, <b>{st.session_state["patient_name"]}</b><br>'
        f'<span style="font-size:0.78rem; color:#93a5c4;">Patient ID: #{st.session_state["patient_id"]}</span></div>',
        unsafe_allow_html=True
    )

    st.markdown("### Menu")
    nav_options = ["🏠 Dashboard", "📅 Book Appointment", "📋 Medical History", "💬 Ask Assistant", "🚨 Emergency"]
    current_index = nav_options.index(st.session_state["view"]) if st.session_state["view"] in nav_options else 0
    nav_choice = st.radio(
        "Navigate",
        nav_options,
        index=current_index,
        label_visibility="collapsed",
        key="nav_radio"
    )
    st.session_state["view"] = nav_choice

    st.markdown(
        '<div style="background: rgba(56, 189, 248, 0.12); border: 1px solid rgba(56, 189, 248, 0.4); '
        'border-radius: 10px; padding: 10px 14px; margin-top: 10px; font-size: 0.82rem; color: #bae6fd;">'
        '💡 <b>Tip:</b> Click <b>💬 Ask Assistant</b> to chat freely with our AI — ask anything, '
        'book appointments in plain language, or get instant answers.</div>',
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state["patient_id"] = None
        st.session_state["patient_name"] = None
        st.session_state["messages"] = []
        st.session_state["pending_appointment"] = None
        st.rerun()

# ------------------ HEADER (shown on every page) ------------------
st.markdown("""
<div class="app-header">
    <div class="icon-box">🏥</div>
    <div class="app-title">MediCare AI Assistant</div>
</div>
""", unsafe_allow_html=True)

view = st.session_state["view"]

# ==================================================================
# VIEW: DASHBOARD
# ==================================================================
if view == "🏠 Dashboard":
    appointments = get_appointments_by_patient(st.session_state["patient_id"]) or []
    upcoming = [a for a in appointments if a["appointment_date"] >= date.today() and a["status"] != "Cancelled"]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-number">{len(appointments)}</div><div class="metric-label">Total Appointments</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-number">{len(upcoming)}</div><div class="metric-label">Upcoming</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-number">{len(appointments) - len(upcoming)}</div><div class="metric-label">Past / Cancelled</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Quick Actions")
    qc1, qc2, qc3 = st.columns(3)
    with qc1:
        if st.button("📅 Book an Appointment", use_container_width=True):
            st.session_state["view"] = "📅 Book Appointment"
            st.rerun()
    with qc2:
        if st.button("📋 View Medical History", use_container_width=True):
            st.session_state["view"] = "📋 Medical History"
            st.rerun()
    with qc3:
        if st.button("💬 Ask a Question", use_container_width=True):
            st.session_state["view"] = "💬 Ask Assistant"
            st.rerun()

    if upcoming:
        st.markdown("#### Your Next Appointment")
        nxt = sorted(upcoming, key=lambda a: (a["appointment_date"], a["appointment_time"]))[0]
        st.info(f"**Dr. {nxt['doctor_name']}** ({nxt['specialization']}) — {nxt['appointment_date']} at {nxt['appointment_time']}")

# ==================================================================
# VIEW: BOOK APPOINTMENT
# ==================================================================
elif view == "📅 Book Appointment":
    st.markdown("### Book an Appointment")

    doctors = get_all_doctors() or []
    if not doctors:
        st.warning("No doctors are available in the system right now.")
    else:
        doctor_labels = [f"Dr. {d['full_name']} — {d['specialization']}" for d in doctors]
        selected_label = st.selectbox("Choose a Doctor", doctor_labels)
        selected_doctor = doctors[doctor_labels.index(selected_label)]

        col1, col2 = st.columns(2)
        with col1:
            appt_date = st.date_input("Preferred Date", min_value=date.today())
        with col2:
            appt_time = st.time_input("Preferred Time")

        reason = st.text_area("Reason for visit", placeholder="e.g. Fever and headache for 2 days")

        if st.button("Book Appointment", use_container_width=True, type="primary"):
            time_str = appt_time.strftime("%H:%M")
            if is_slot_available(selected_doctor["doctor_id"], appt_date, time_str):
                appt_id = create_appointment(
                    st.session_state["patient_id"],
                    selected_doctor["doctor_id"],
                    appt_date,
                    time_str,
                    reason.strip()
                )
                st.success(f"✅ Appointment booked! Confirmation ID #{appt_id} with Dr. {selected_doctor['full_name']} on {appt_date} at {time_str}.")
            else:
                next_slot = get_next_available_slot(selected_doctor["doctor_id"], appt_date, time_str)
                if next_slot:
                    st.warning(f"That slot is already booked. Next available slot on {appt_date} is **{next_slot}**.")
                    if st.button(f"Book {next_slot} instead", key="book_next_slot"):
                        appt_id = create_appointment(
                            st.session_state["patient_id"],
                            selected_doctor["doctor_id"],
                            appt_date,
                            next_slot,
                            reason.strip()
                        )
                        st.success(f"✅ Appointment booked! Confirmation ID #{appt_id} at {next_slot}.")
                else:
                    st.error("No free slots left with this doctor on that day. Please choose another date.")

# ==================================================================
# VIEW: MEDICAL HISTORY
# ==================================================================
elif view == "📋 Medical History":
    st.markdown("### Your Appointment History")
    appointments = get_appointments_by_patient(st.session_state["patient_id"]) or []

    if not appointments:
        st.info("You don't have any appointments yet.")
    else:
        table_data = [
            {
                "Date": a["appointment_date"],
                "Time": a["appointment_time"],
                "Doctor": f"Dr. {a['doctor_name']}",
                "Specialization": a["specialization"],
                "Reason": a["reason"],
                "Status": a["status"],
            }
            for a in appointments
        ]
        st.dataframe(table_data, use_container_width=True, hide_index=True)

# ==================================================================
# VIEW: ASK ASSISTANT (general chat)
# ==================================================================
elif view == "💬 Ask Assistant":
    st.markdown("### Ask the Medical AI Assistant")

    if not st.session_state["messages"]:
        with st.chat_message("assistant", avatar="🏥"):
            st.markdown("Hi! Ask me any general health question, or I can help route you to booking or history.")

    for msg in st.session_state["messages"]:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        avatar = "🧑" if role == "user" else "🏥"
        with st.chat_message(role, avatar=avatar):
            st.markdown(msg.content)

    user_input = st.chat_input("Type your message here...")
    if user_input:
        st.session_state["messages"].append(HumanMessage(content=user_input))
        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_input)

        state = {
            "messages": st.session_state["messages"],
            "patient_id": st.session_state.get("patient_id"),
            "intent": None,
            "next_agent": None,
            "is_emergency": False,
            "pending_appointment": st.session_state.get("pending_appointment"),
        }
        with st.chat_message("assistant", avatar="🏥"):
            with st.spinner("Thinking..."):
                result = st.session_state["graph_app"].invoke(state)
                ai_response = result["messages"][-1].content
                st.markdown(ai_response)
        st.session_state["messages"] = result["messages"]
        st.session_state["pending_appointment"] = result.get("pending_appointment")

    if st.button("🗑️ Clear Conversation"):
        st.session_state["messages"] = []
        st.session_state["pending_appointment"] = None
        st.rerun()

# ==================================================================
# VIEW: EMERGENCY
# ==================================================================
elif view == "🚨 Emergency":
    st.markdown("""
    <div class="emergency-banner">
        🚨 <b>If this is a life-threatening emergency, call your local emergency number immediately.</b><br>
        This assistant can help flag urgent symptoms, but it is not a substitute for emergency medical services.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Describe what's happening")

    if not st.session_state["messages"]:
        with st.chat_message("assistant", avatar="🏥"):
            st.markdown("Please describe your symptoms or the emergency in detail. I'll assess and flag it appropriately.")

    for msg in st.session_state["messages"]:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        avatar = "🧑" if role == "user" else "🏥"
        with st.chat_message(role, avatar=avatar):
            st.markdown(msg.content)

    user_input = st.chat_input("Describe the emergency...")
    if user_input:
        st.session_state["messages"].append(HumanMessage(content=user_input))
        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_input)

        state = {
            "messages": st.session_state["messages"],
            "patient_id": st.session_state.get("patient_id"),
            "intent": "emergency",
            "next_agent": None,
            "is_emergency": True,
            "pending_appointment": st.session_state.get("pending_appointment"),
        }
        with st.chat_message("assistant", avatar="🏥"):
            with st.spinner("Assessing..."):
                result = st.session_state["graph_app"].invoke(state)
                ai_response = result["messages"][-1].content
                st.markdown(ai_response)
        st.session_state["messages"] = result["messages"]
        st.session_state["pending_appointment"] = result.get("pending_appointment")