from database.connection import get_connection


# ------------------ PATIENT FUNCTIONS ------------------

def create_patient(full_name, gender, date_of_birth, phone, address):
    """Adds a new patient"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO patients (full_name, gender, date_of_birth, phone, address)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING patient_id;
        """,
        (full_name, gender, date_of_birth, phone, address)
    )
    patient_id = cursor.fetchone()["patient_id"]
    conn.commit()
    cursor.close()
    conn.close()
    return patient_id


def get_patient_by_id(patient_id):
    """Fetches patient details by id"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patients WHERE patient_id = %s;", (patient_id,))
    patient = cursor.fetchone()
    cursor.close()
    conn.close()
    return patient


def get_patient_by_name(full_name):
    """Searches for a patient by name"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patients WHERE full_name ILIKE %s;", (f"%{full_name}%",))
    patients = cursor.fetchall()
    cursor.close()
    conn.close()
    return patients


# ------------------ DOCTOR FUNCTIONS ------------------

def get_all_doctors():
    """Returns the list of all doctors"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM doctors;")
    doctors = cursor.fetchall()
    cursor.close()
    conn.close()
    return doctors


def get_doctors_by_specialization(specialization):
    """Finds doctors by specialization"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM doctors WHERE specialization ILIKE %s;",
        (f"%{specialization}%",)
    )
    doctors = cursor.fetchall()
    cursor.close()
    conn.close()
    return doctors


# ------------------ APPOINTMENT FUNCTIONS ------------------

def create_appointment(patient_id, doctor_id, appointment_date, appointment_time, reason):
    """Books a new appointment"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO appointments (patient_id, doctor_id, appointment_date, appointment_time, reason)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING appointment_id;
        """,
        (patient_id, doctor_id, appointment_date, appointment_time, reason)
    )
    appointment_id = cursor.fetchone()["appointment_id"]
    conn.commit()
    cursor.close()
    conn.close()
    return appointment_id


def get_appointments_by_patient(patient_id):
    """Fetches all appointments for a given patient"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT a.*, d.full_name AS doctor_name, d.specialization
        FROM appointments a
        JOIN doctors d ON a.doctor_id = d.doctor_id
        WHERE a.patient_id = %s
        ORDER BY a.appointment_date DESC;
        """,
        (patient_id,)
    )
    appointments = cursor.fetchall()
    cursor.close()
    conn.close()
    return appointments


# ------------------ SLOT AVAILABILITY FUNCTIONS ------------------

def is_slot_available(doctor_id, appointment_date, appointment_time):
    """Checks whether a specific slot is free for a doctor"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT * FROM appointments
        WHERE doctor_id = %s AND appointment_date = %s AND appointment_time = %s
        AND status != 'Cancelled';
        """,
        (doctor_id, appointment_date, appointment_time)
    )
    existing = cursor.fetchone()
    cursor.close()
    conn.close()
    return existing is None


def get_next_available_slot(doctor_id, appointment_date, requested_time, slot_gap_minutes=30):
    """
    If the requested time is busy, finds the next free slot on the same day
    """
    from datetime import datetime, timedelta

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT appointment_time FROM appointments
        WHERE doctor_id = %s AND appointment_date = %s AND status != 'Cancelled'
        ORDER BY appointment_time;
        """,
        (doctor_id, appointment_date)
    )
    booked_times = [row["appointment_time"] for row in cursor.fetchall()]
    cursor.close()
    conn.close()

    # Starting from requested_time, look for the next free slot based on the gap
    current_time = datetime.strptime(str(requested_time), "%H:%M:%S") if len(str(requested_time)) > 5 else datetime.strptime(str(requested_time), "%H:%M")

    # Clinic hours: 8 AM to 5 PM (adjustable)
    end_time = current_time.replace(hour=17, minute=0)

    while current_time <= end_time:
        current_time_only = current_time.time()
        if current_time_only not in booked_times:
            return current_time.strftime("%H:%M")
        current_time += timedelta(minutes=slot_gap_minutes)

    return None  # No free slot found on that day

def get_patient_by_phone(phone):
    """Searches for a patient by phone number"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patients WHERE phone = %s;", (phone,))
    patient = cursor.fetchone()
    cursor.close()
    conn.close()
    return patient


def phone_already_registered(phone):
    """Checks if a phone number is already registered (to prevent duplicate accounts)"""
    return get_patient_by_phone(phone) is not None


def get_todays_appointments(doctor_id=None):
    """Fetches today's appointments, optionally filtered by doctor"""
    from datetime import date
    conn = get_connection()
    cursor = conn.cursor()

    if doctor_id:
        cursor.execute(
            """
            SELECT a.*, p.full_name AS patient_name, p.phone AS patient_phone,
                   d.full_name AS doctor_name
            FROM appointments a
            JOIN patients p ON a.patient_id = p.patient_id
            JOIN doctors d ON a.doctor_id = d.doctor_id
            WHERE a.appointment_date = %s AND a.doctor_id = %s
            AND a.status != 'Cancelled'
            ORDER BY a.appointment_time;
            """,
            (date.today(), doctor_id)
        )
    else:
        cursor.execute(
            """
            SELECT a.*, p.full_name AS patient_name, p.phone AS patient_phone,
                   d.full_name AS doctor_name
            FROM appointments a
            JOIN patients p ON a.patient_id = p.patient_id
            JOIN doctors d ON a.doctor_id = d.doctor_id
            WHERE a.appointment_date = %s
            AND a.status != 'Cancelled'
            ORDER BY a.appointment_time;
            """,
            (date.today(),)
        )

    appointments = cursor.fetchall()
    cursor.close()
    conn.close()
    return appointments