import os
from dotenv import load_dotenv

load_dotenv()

CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING")
TABLE_NAME = os.getenv("TABLE_NAME")

connection_pool = None
