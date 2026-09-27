import streamlit as st
import base64
from langchain_core.messages import HumanMessage, AIMessage
from graph.build_graph import build_graph
from database.models import get_patient_by_id

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="MediCare AI | Medical Assistant",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ------------------ BACKGROUND + FONTS + THEME ------------------
def set_background(image_path):
    try:
        with open(image_path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        bg_css = f"""
            background: linear-gradient(180deg, rgba(6, 20, 45, 0.72) 0%, rgba(6, 20, 45, 0.85) 100%),
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
        color: #e6ecf5;}}

       [data-testid="stIconMaterial"], .material-icons, span[class*="material-icons"] {{
       font-family: 'Material Symbols Rounded', 'Material Icons' !important;
       font-weight: normal !important;
       font-style: normal !important;}}

        .stApp {{
            {bg_css}
        }}

        /* Hide default Streamlit chrome */
        #MainMenu, footer {{visibility: hidden;}}

        /* ---------- HEADER ---------- */
        .app-header {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 14px;
            padding: 28px 0 6px 0;
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
            margin-bottom: 6px;
        }}
        .pill-row {{
            display: flex;
            justify-content: center;
            gap: 10px;
            flex-wrap: wrap;
            margin: 14px 0 28px 0;
        }}
        .pill {{
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.12);
            color: #cbd5e1;
            padding: 6px 14px;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 500;
            backdrop-filter: blur(6px);
        }}

        /* ---------- CHAT BUBBLES ---------- */
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

        /* ---------- CHAT INPUT ---------- */
        .stChatInputContainer, [data-testid="stChatInput"] {{
            background: rgba(255,255,255,0.06) !important;
            border: 1px solid rgba(255,255,255,0.15) !important;
            border-radius: 14px !important;
        }}
        [data-testid="stChatInput"] textarea {{
            color: #1a1a1a !important;
            background-color: #ffffff !important;
        }}
        [data-testid="stChatInput"] textarea::placeholder {{
            color: #6b7280 !important;
        }}

        /* ---------- SIDEBAR ---------- */
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
        .sidebar-title {{
            font-family: 'Poppins', sans-serif;
            font-weight: 600;
            font-size: 1rem;
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 4px;
        }}
        .status-dot {{
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 6px;
        }}
        .status-online {{ background: #22c55e; box-shadow: 0 0 8px #22c55e; }}
        .status-offline {{ background: #f59e0b; box-shadow: 0 0 8px #f59e0b; }}

        section[data-testid="stSidebar"] input {{
            background: #ffffff !important;
            border-radius: 8px !important;
            border: 1px solid rgba(255,255,255,0.15) !important;
            color: #1a1a1a !important;}}

        section[data-testid="stSidebar"] input::placeholder {{
            color: #6b7280 !important;}}

        section[data-testid="stSidebar"] button {{
            background: rgba(239, 68, 68, 0.12) !important;
            border: 1px solid rgba(239, 68, 68, 0.4) !important;
            border-radius: 8px !important;
            color: #fca5a5 !important;
            font-weight: 500 !important;
        }}
    </style>
    """, unsafe_allow_html=True)

set_background("assets/background.jpg")

# ------------------ SIDEBAR ------------------
from database.models import get_patient_by_id, get_patient_by_phone, phone_already_registered, create_patient

# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.markdown('<div class="sidebar-title">🏥 Patient Access</div>', unsafe_allow_html=True)

    # Initialize session state
    if "patient_id" not in st.session_state:
        st.session_state["patient_id"] = None
    if "patient_name" not in st.session_state:
        st.session_state["patient_name"] = None

    # ---- IF ALREADY LOGGED IN ----
    if st.session_state["patient_id"]:
        st.markdown(
            f'<div class="sidebar-card"><span class="status-dot status-online"></span>'
            f'Welcome, <b>{st.session_state["patient_name"]}</b>!<br>'
            f'<span style="font-size:0.78rem; color:#93a5c4;">Patient ID: #{st.session_state["patient_id"]}</span></div>',
            unsafe_allow_html=True
        )
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state["patient_id"] = None
            st.session_state["patient_name"] = None
            st.session_state["messages"] = []
            st.rerun()

    # ---- IF NOT LOGGED IN: SHOW LOGIN / REGISTER TABS ----
    else:
        tab1, tab2 = st.tabs(["🔑 Login", "📝 New Patient"])

        # ---- LOGIN TAB ----
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

        # ---- REGISTER TAB ----
        with tab2:
            reg_name = st.text_input("Full Name", key="reg_name")
            reg_gender = st.selectbox("Gender", ["Male", "Female", "Other"], key="reg_gender")
            reg_dob = st.date_input("Date of Birth", key="reg_dob", min_value=None)
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
                    st.success(f"Registered successfully! Welcome, {reg_name.strip()}.")
                    st.rerun()

    st.markdown('<div class="sidebar-title" style="margin-top:22px;">ℹ️ What I Can Do</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-card">
    📅 &nbsp;Book appointments<br><br>
    📋 &nbsp;View medical history<br><br>
    💬 &nbsp;Answer health questions<br><br>
    🚨 &nbsp;Detect emergencies
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑️  Clear Conversation", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()
# ------------------ HEADER ------------------
st.markdown("""
<div class="app-header">
    <div class="icon-box">🏥</div>
    <div class="app-title">MediCare AI Assistant</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="app-subtitle">Your intelligent multi-agent healthcare companion</div>', unsafe_allow_html=True)

st.markdown("""
<div class="pill-row">
    <div class="pill">📅 Appointments</div>
    <div class="pill">📋 Medical History</div>
    <div class="pill">🚨 Emergency Detection</div>
    <div class="pill">💬 General Queries</div>
</div>
""", unsafe_allow_html=True)

# ------------------ SESSION STATE ------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "graph_app" not in st.session_state:
    st.session_state["graph_app"] = build_graph()

# ------------------ WELCOME MESSAGE (only if chat is empty) ------------------
if not st.session_state["messages"]:
    with st.chat_message("assistant", avatar="🏥"):
        st.markdown(
            "Hello! I'm your Medical AI Assistant. I can help you **book appointments**, "
            "**check your medical history**, answer **general health questions**, or "
            "flag **emergencies**. How can I help you today?"
        )

# ------------------ DISPLAY CHAT HISTORY ------------------
for msg in st.session_state["messages"]:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    avatar = "🧑" if role == "user" else "🏥"
    with st.chat_message(role, avatar=avatar):
        st.markdown(msg.content)

# ------------------ CHAT INPUT ------------------
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
        "is_emergency": False
    }

    with st.chat_message("assistant", avatar="🏥"):
        with st.spinner("Thinking..."):
            result = st.session_state["graph_app"].invoke(state)
            ai_response = result["messages"][-1].content
            st.markdown(ai_response)

    st.session_state["messages"] = result["messages"]