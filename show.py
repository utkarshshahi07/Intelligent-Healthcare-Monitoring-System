import sqlite3


conn = sqlite3.connect('incident_logs.db')
cursor = conn.cursor()


cursor.execute('SELECT * FROM incidents')
rows = cursor.fetchall()


for row in rows:
    print(row)

conn.close()
