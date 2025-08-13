import sqlite3


conn = sqlite3.connect('face_recognition.db')
cursor = conn.cursor()


cursor.execute('SELECT * FROM PersonData')
rows = cursor.fetchall()


for row in rows:
    print(row)

conn.close()
