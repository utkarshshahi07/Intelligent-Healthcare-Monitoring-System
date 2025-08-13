import cv2
import face_recognition
import sqlite3
import random
import os
import numpy as np

# Initialize the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

# Connect to SQLite database
db_connection = sqlite3.connect("face_recognition.db")
db_cursor = db_connection.cursor()

# Create the PersonData table if it doesn't exist
db_cursor.execute("""
CREATE TABLE IF NOT EXISTS PersonData (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    behavior_score INTEGER,
    image_path TEXT
)
""")
db_connection.commit()

# Function to check if the face already exists in the database
def is_face_in_database(face_encoding):
    db_cursor.execute("SELECT image_path FROM PersonData")
    rows = db_cursor.fetchall()
    
    for row in rows:
        # Load the saved image's face encoding and compare it
        saved_image_path = row[0]
        saved_image = face_recognition.load_image_file(saved_image_path)
        saved_encoding = face_recognition.face_encodings(saved_image)
        
        if saved_encoding:
            saved_encoding = saved_encoding[0]
            # Compare the face encodings
            match = face_recognition.compare_faces([saved_encoding], face_encoding)
            if True in match:
                return True  # Face is already in the database
    return False

# Function to insert the data into the database
def insert_person_into_db(image_path):
    """Insert a person into the database with random behavior score and image path."""
    name = "Unknown"
    behavior_score = 100  # Generate a random behavior score
    
    db_cursor.execute("INSERT INTO PersonData (name, behavior_score, image_path) VALUES (?, ?, ?)", 
                      (name, behavior_score, image_path))
    db_connection.commit()
    print(f"Added Unknown to the database with behavior score: {behavior_score} and image path: {image_path}")

# Function to capture and store face image
def capture_and_store_face():
    print("Press 'q' to exit the program.")
    
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        # Convert the frame to RGB (face_recognition format)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # bgr to rgb

        # Detect faces in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Check if the detected face is already in the database
            if is_face_in_database(face_encoding):
                print("Face already recognized in the database. Skipping capture.")
            else:
                # If not recognized, capture the face image
                top, right, bottom, left = face_location
                face_image = frame[top:bottom, left:right]  # Capture the face region
                image_path = f"unknown_{random.randint(1000, 9999)}.jpg"  # Random file name
                cv2.imwrite(image_path, face_image)  # Save the captured face as an image

                # Insert the new person into the database
                insert_person_into_db(image_path)
                print(f"Captured face saved as {image_path}")

        # Display the frame with detected faces
        cv2.imshow("Face Recognition", frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    db_connection.close()

# Run the capture and store function
capture_and_store_face()