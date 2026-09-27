import os
import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv

load_dotenv()


def send_appointment_email(doctor_email, doctor_name, patient_name, appointment_date, appointment_time, reason):
    """
    Sends an email notification to the doctor when a new appointment is booked
    """
    sender_email = os.getenv("EMAIL_ADDRESS")
    sender_password = os.getenv("EMAIL_APP_PASSWORD")

    subject = f"New Appointment Booked - {appointment_date}"
    body = f"""
Dear Dr. {doctor_name},

A new appointment has been booked with you.

Patient Name: {patient_name}
Date: {appointment_date}
Time: {appointment_time}
Reason: {reason}

Please check your dashboard for more details.

- MediCare AI Assistant
"""

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = doctor_email

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, doctor_email, msg.as_string())
        print("✅ Email sent successfully to doctor.")
        return True
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
        return False