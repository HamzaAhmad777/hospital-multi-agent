import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

def get_connection():
    """
    Creates a connection to the PostgreSQL database
    """
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            cursor_factory=RealDictCursor
        )
        return conn
    except Exception as e:
        print(f"❌ Database connection error: {e}")
        return None


def test_connection():
    """
    Function to test the connection
    """
    conn = get_connection()
    if conn:
        print("✅ Database connection successful!")
        conn.close()
    else:
        print("❌ Could not connect to the database.")


if __name__ == "__main__":
    test_connection()