#!/bin/bash

# Define the connection string for clarity
DB_CONN_STRING="postgresql://app_cross_dataset_discovery:64P5EiXwdDyKBtTuG1h68Jk1AqeB@172.16.59.6:30432/db_cross_dataset_discovery"

# The SQL command to execute. Using a "here document" (<<EOF) for readability.
psql "$DB_CONN_STRING" <<EOF
-- Set a higher memory allocation for this session to speed up the index build.
-- 1GB is a good value if the server has enough RAM. Adjust if needed.
SET maintenance_work_mem = '512MB';

-- Display the time it takes to run the command
\timing

-- Create the index. This is the long-running command.
-- Using CONCURRENTLY is crucial for a live production database.
CREATE INDEX CONCURRENTLY idx_gin_ts_content_mathe ON embeddings_cross_dataset_discovery_mathe_language USING GIN(ts_content);

-- The script will wait here until the command above is finished.
EOF

# This will only be printed after the psql command finishes successfully
echo "Index creation command finished at $(date)"
