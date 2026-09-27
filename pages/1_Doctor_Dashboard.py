import streamlit as st
from database.models import get_all_doctors, get_todays_appointments

st.set_page_config(page_title="Doctor Dashboard", page_icon="👨‍⚕️", layout="centered")

st.markdown("""
<style>
    .stApp { background: radial-gradient(circle at 20% 10%, #0f2a4a 0%, #061426 55%, #030a16 100%); }
    h1, h2, h3, p, span, div { color: #e6ecf5; }
    .appt-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #08152b 0%, #050d1c 100%) !important;
    }
    section[data-testid="stSidebar"] * {
        color: #dbe4f3 !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("👨‍⚕️ Doctor Dashboard")
st.markdown("View today's scheduled appointments")
st.markdown("---")

doctors = get_all_doctors()
doctor_names = {f"{d['full_name']} ({d['specialization']})": d["doctor_id"] for d in doctors}

selected = st.selectbox("Select Doctor", list(doctor_names.keys()))
doctor_id = doctor_names[selected]

st.markdown("### 📅 Today's Appointments")

appointments = get_todays_appointments(doctor_id=doctor_id)

if not appointments:
    st.info("No appointments scheduled for today.")
else:
    for appt in appointments:
        st.markdown(f"""
        <div class="appt-card">
            <b>🕐 {appt['appointment_time']}</b><br>
            👤 Patient: <b>{appt['patient_name']}</b><br>
            📞 Phone: {appt['patient_phone']}<br>
            📝 Reason: {appt['reason']}<br>
            📌 Status: {appt['status']}
        </div>
        """, unsafe_allow_html=True)