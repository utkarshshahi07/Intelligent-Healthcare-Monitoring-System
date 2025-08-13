import sqlite3

# Connect to the SQLite database
conn = sqlite3.connect('incident_logs.db')
cursor = conn.cursor()

# Get all the table names
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

# Loop through each table and delete all data
for table in tables:
    cursor.execute(f"DELETE FROM {table[0]};")
    print(f"Deleted all data from {table[0]}")

# Commit the transaction
conn.commit()

# Close the connection
conn.close()
