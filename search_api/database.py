import os
import psycopg2
from psycopg2.pool import SimpleConnectionPool
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASS"),
}
TABLE_NAME = os.getenv("TABLE_NAME")

connection_pool = SimpleConnectionPool(minconn=1, maxconn=10, **DB_CONFIG)