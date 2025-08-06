import os
import psycopg2
from dotenv import load_dotenv


def check_database_connection():
    """
    Loads database credentials from a .env file, connects to the PostgreSQL
    database, and verifies the connection by fetching the server version.
    """
    # Load environment variables from the .env file
    load_dotenv()

    # Retrieve database connection details from environment variables
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT")
    db_name = os.getenv("DB_NAME")
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")

    # Check if all required environment variables are set
    if not all([db_host, db_port, db_name, db_user, db_password]):
        print("Error: One or more database environment variables are missing.")
        print("Please check your .env file.")
        return

    conn = None
    try:
        # --- 1. Attempt to connect to the database ---
        print(
            f"Attempting to connect to database '{db_name}' on {db_host}:{db_port}..."
        )
        conn = psycopg2.connect(
            host=db_host,
            port=db_port,
            dbname=db_name,
            user=db_user,
            password=db_password,
        )

        # --- 2. If connection is successful, verify it with a simple query ---
        # Create a cursor object
        cur = conn.cursor()

        # Execute a simple query to get the PostgreSQL version
        cur.execute("SELECT version();")

        # Fetch the result
        db_version = cur.fetchone()

        print("\n✅ Success! Database connection is working.")
        print(f"PostgreSQL Version: {db_version[0]}")

        # Close the cursor
        cur.close()

    except psycopg2.Error as e:
        # --- 3. Catch any database errors and print them ---
        print("\n❌ Failed to connect to the database.")
        print(f"Error: {e}")

    finally:
        # --- 4. Always close the connection if it was opened ---
        if conn is not None:
            conn.close()
            print("\nConnection closed.")


if __name__ == "__main__":
    check_database_connection()
