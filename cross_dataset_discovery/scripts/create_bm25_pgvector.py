import psycopg2
import os

DB_HOST = "172.16.59.6"
DB_PORT = "30432"
DB_NAME = "db_cross_dataset_discovery"
DB_USER = "app_cross_dataset_discovery"
DB_PASS = "64P5EiXwdDyKBtTuG1h68Jk1AqeB"
TABLE_NAME = "embeddings_cross_dataset_discovery_mathe_language"
TEXT_COLUMN = "content"
TSVECTOR_COLUMN = "ts_content"
INDEX_NAME = f"idx_{TSVECTOR_COLUMN}_gin"
print("Creating Full-Text Search index for BM25...")
conn_string = f"dbname='{DB_NAME}' user='{DB_USER}' host='{DB_HOST}' port='{DB_PORT}' password='{DB_PASS}'"

conn = psycopg2.connect(conn_string)
conn.autocommit = True
cursor = conn.cursor()
print("Connected to the database.")
sql_add_column = f"""
ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS {TSVECTOR_COLUMN} TSVECTOR;
"""
cursor.execute(sql_add_column)
print(f"Added column {TSVECTOR_COLUMN} to table {TABLE_NAME} if it did not exist.")
sql_update_column = f"""
UPDATE {TABLE_NAME} SET {TSVECTOR_COLUMN} = to_tsvector('english', {TEXT_COLUMN});
"""
cursor.execute(sql_update_column)
print(f"Updated column {TSVECTOR_COLUMN} in table {TABLE_NAME} with tsvector data from {TEXT_COLUMN}.")
sql_create_index = f"""
CREATE INDEX IF NOT EXISTS {INDEX_NAME} ON {TABLE_NAME} USING GIN({TSVECTOR_COLUMN});
"""
cursor.execute(sql_create_index)

cursor.close()
conn.close()

print("One-time Full-Text Search index build is complete.")