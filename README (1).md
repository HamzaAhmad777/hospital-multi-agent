# 🏥 MediCare AI Assistant — Medical AI Multi-Agent System

A patient-facing Medical AI Assistant built using **LangGraph** multi-agent architecture. The system understands patient queries, routes them to the correct specialized agent, and responds using real hospital data — including live appointment scheduling, medical history, emergency detection, and a Retrieval-Augmented Generation (RAG) knowledge base for general hospital queries.

---

## 1. Objective

Traditional healthcare systems are slow and impersonal. This project builds an AI system that:
- Interacts with patients in real time through a chat interface
- Understands context (including patient history)
- Makes routing decisions automatically based on intent
- Handles different types of tasks (appointments, history, emergencies, general queries) through separate specialized agents

---

## 2. System Architecture

The system is built as a **LangGraph state graph**. Every conversation follows this flow:

```
                          ┌─────────────────┐
                          │   User Input     │
                          └────────┬─────────┘
                                   │
                                   ▼
                          ┌─────────────────┐
                          │   Supervisor     │  (Intent Classifier)
                          │   Agent (LLM)    │
                          └────────┬─────────┘
                                   │
              ┌────────────┬───────┴────────┬────────────────┐
              ▼            ▼                ▼                ▼
      ┌──────────────┐ ┌──────────┐ ┌────────────────┐ ┌──────────────┐
      │  Emergency    │ │Appointment│ │ Medical History│ │   General     │
      │    Agent      │ │  Agent    │ │     Agent      │ │  Query Agent  │
      └──────┬────────┘ └─────┬─────┘ └───────┬────────┘ └───────┬───────┘
             │                │               │                  │
             │                ▼               ▼                  ▼
             │        ┌────────────────────────────┐    ┌────────────────┐
             │        │   PostgreSQL Database        │    │  RAG Pipeline   │
             │        │ (patients/doctors/appts)     │    │ (ChromaDB-style │
             │        └────────────────────────────┘    │  vector search) │
             │                                            └────────────────┘
             └────────────────────┬──────────────────────────────┘
                                   ▼
                          ┌─────────────────┐
                          │  Final Response  │
                          └─────────────────┘
```

### 2.1 Router (Supervisor / Intent Classifier)
The **Supervisor Agent** receives every incoming patient message and uses an LLM (Groq — Llama 3.3 70B) to classify it into one of four categories:
- `emergency`
- `appointment`
- `medical_history`
- `general`

Based on this classification, LangGraph's **conditional edges** route the state to the correct specialized agent node.

### 2.2 Specialized Agents

| Agent | Responsibility |
|---|---|
| **Emergency Agent** | Detects life-threatening situations (chest pain, breathing difficulty, bleeding, etc.) and immediately returns safe, non-diagnostic emergency guidance (e.g., directs to ER / ambulance number). Never attempts to give medical advice in emergencies. |
| **Appointment Agent** | Extracts specialization, date, time, and reason from the patient's message using an LLM, checks doctor availability against the database, prevents double-booking, and suggests the next available slot if the requested time is taken. Sends an email notification to the doctor once booked. |
| **Medical History Agent** | Retrieves the logged-in patient's past appointments (with doctor name, date, time, reason, and status) from PostgreSQL. |
| **General Query Agent** | Answers general health and hospital-policy questions using a **RAG pipeline** — retrieves relevant excerpts from the hospital's internal knowledge base before generating a grounded answer. |

### 2.3 Memory / Database Layer
All patient, doctor, and appointment data is stored in **PostgreSQL** (`hospital_db`), with three core tables:
- `patients` — patient_id, full_name, gender, date_of_birth, phone, address
- `doctors` — doctor_id, full_name, specialization, phone, email
- `appointments` — appointment_id, patient_id (FK), doctor_id (FK), date, time, reason, status

A `UNIQUE` constraint style check (`is_slot_available`) prevents two patients from booking the same doctor at the same date/time.

### 2.4 RAG (Retrieval-Augmented Generation) Pipeline
Used by the General Query Agent to ground answers in real hospital policy rather than generic LLM knowledge.

**Ingestion (one-time):**
1. A hospital knowledge document (`knowledge_base/hospital_info.txt`) containing policies (operating hours, fasting requirements, cancellation policy, emergency procedures, etc.) is split into paragraph-level chunks.
2. Each chunk is converted into a vector embedding using `sentence-transformers` (`all-MiniLM-L6-v2`).
3. Chunks and their embeddings are stored in a local JSON-based vector store (`knowledge_base/vector_store.json`).

**Query time:**
1. The patient's question is embedded using the same model.
2. Cosine similarity is computed between the query embedding and all stored chunk embeddings.
3. The top 3 most relevant chunks are retrieved (plain text, not vectors).
4. These chunks are injected into the LLM prompt as context.
5. The LLM generates an answer grounded in the retrieved hospital-specific information.

> Note: ChromaDB was originally used for this pipeline but was replaced with a lightweight custom implementation (NumPy cosine similarity) due to a native DLL dependency (`grpc`) being blocked by the local machine's Application Control security policy. The retrieval logic and RAG concept remain identical — only the vector storage backend changed.

---

## 3. Technology Stack

| Component | Technology |
|---|---|
| Multi-agent orchestration | LangGraph |
| LLM Provider | Groq (Llama 3.3 70B / GPT-OSS 120B) |
| Database | PostgreSQL |
| DB Driver | psycopg2 |
| Vector Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Frontend | Streamlit |
| Email Notifications | smtplib (Gmail SMTP) |
| Environment Config | python-dotenv |

---

## 4. Project Structure

```
hospital_multi_agent/
│
├── main.py                     # CLI chat interface
├── app.py                      # Streamlit web interface (patient-facing)
├── .env                        # API keys & DB credentials (not committed)
│
├── database/
│   ├── connection.py            # PostgreSQL connection handler
│   ├── models.py                # CRUD functions (patients, doctors, appointments)
│   └── vector_store.py          # RAG ingestion + retrieval (embeddings, cosine similarity)
│
├── agents/
│   ├── supervisor_agent.py      # Intent classification / routing
│   ├── emergency_agent.py       # Emergency detection & safe response
│   ├── appointment_agent.py     # Appointment booking with slot-conflict handling
│   ├── medical_history_agent.py # Patient appointment history retrieval
│   └── general_query_agent.py   # RAG-powered general/health Q&A
│
├── graph/
│   ├── state.py                  # Shared LangGraph state schema
│   └── build_graph.py            # Graph construction (nodes, edges, routing)
│
├── pages/
│   └── 1_Doctor_Dashboard.py     # Doctor-facing view of today's appointments
│
├── knowledge_base/
│   ├── hospital_info.txt         # Source document for RAG
│   └── vector_store.json         # Generated embeddings (after ingestion)
│
└── utils/
    └── helpers.py                 # Email notification logic
```

---

## 5. Key Features

- ✅ **Intent-based routing** — every message is classified before being handled
- ✅ **Double-booking prevention** — checks slot availability and suggests the next open time
- ✅ **Emergency safety** — never gives medical advice during emergencies, only directs to real help
- ✅ **RAG-grounded answers** — general queries pull from actual hospital policy, not just LLM guesses
- ✅ **Persistent memory** — full appointment history stored and retrievable per patient
- ✅ **Doctor email notifications** — doctors are notified by email when a new appointment is booked
- ✅ **Doctor Dashboard** — a separate page showing today's appointments per doctor
- ✅ **Patient-friendly authentication** — patients register/login using name and phone number (no need to remember internal IDs)
- ✅ **Professional Streamlit UI** — branded chat interface with session-based conversation memory

---

## 6. Example Test Scenarios

### Test 1 — General Query (RAG)
**Input:** `"Do I need to fast before a blood test?"`
**Flow:** Supervisor → General Query Agent → RAG retrieval → LLM
**Output:** *"For routine tests like a lipid profile or fasting blood-sugar test, you should fast for 8–12 hours before the draw (water is allowed). Most other blood tests do not require fasting."*
*(Answer is grounded in the hospital's actual policy document, not generic knowledge.)*

### Test 2 — Appointment Booking
**Input:** `"I need an appointment with a general physician on 2026-09-26 at 11:00"`
**Flow:** Supervisor → Appointment Agent → DB slot check → Booking created → Email sent to doctor
**Output:** *"✅ Appointment booked successfully! Doctor: Hamza Ahmad (General Physician), Date: 2026-09-26, Time: 11:00, Appointment ID: 3"*

### Test 3 — Double-Booking Prevention
**Input (same slot booked again):** `"I need an appointment with a general physician on 2026-09-26 at 11:00"`
**Flow:** Supervisor → Appointment Agent → Slot conflict detected → Next slot suggested
**Output:** *"❌ The 11:00 slot is already booked with Dr. Hamza Ahmad. ✅ Next available slot: 11:30 on 2026-09-26. Would you like to confirm this time?"*

### Test 4 — Emergency Detection
**Input:** `"I can't breathe properly and my chest hurts badly"`
**Flow:** Supervisor → Emergency Agent (direct response, no LLM medical advice)
**Output:** *"🚨 This looks like an EMERGENCY! Please immediately: 1. Go to the nearest emergency room, or 2. Call an ambulance at 1122 (Pakistan)..."*

### Test 5 — Medical History Retrieval
**Input:** `"Show me my past appointments"`
**Flow:** Supervisor → Medical History Agent → DB query (JOIN with doctors table)
**Output:** *"📋 Medical History for [Patient Name]: • Date: 2026-09-26 | Time: 11:00 | Doctor: Hamza Ahmad (General Physician) | Status: Scheduled"*

---

## 7. Safety & Ethical Considerations

- The system **never provides a medical diagnosis**.
- The system **never recommends specific medicines or dosages**.
- In emergencies, the AI immediately directs the patient to real emergency services rather than attempting to help via conversation.
- Patient data is only accessible after login (via phone number), and no patient can view another patient's records.

---

## 8. How to Run

```bash
# 1. Activate virtual environment
venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Ingest the RAG knowledge base (run once, or after updating hospital_info.txt)
python -m database.vector_store

# 4. Run the Streamlit web app
python -m streamlit run app.py

# OR run the CLI version
python main.py
```

---

## 9. Future Improvements (Not Yet Implemented)

- Voice input/output
- Real hospital API integration
- SMS notifications in addition to email
- Advanced multi-turn conversational memory across sessions (currently memory persists only within a single chat session)
