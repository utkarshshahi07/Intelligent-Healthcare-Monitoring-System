import cv2
import face_recognition
import sqlite3
import numpy as np

# Connect to SQLite database
db_connection = sqlite3.connect("face_recognition.db")
db_cursor = db_connection.cursor()

# Function to load all known faces from the database
def load_known_faces():
    db_cursor.execute("SELECT image_path FROM PersonData")
    rows = db_cursor.fetchall()

    known_faces = []
    known_ids = []

    for row in rows:
        image_path = row[0]
        if image_path:
            try:
                # Load the image from the path and encode the face
                image = face_recognition.load_image_file(image_path)
                face_encoding = face_recognition.face_encodings(image)
                if face_encoding:
                    known_faces.append(face_encoding[0])
                    # Storing the ID to match later
                    db_cursor.execute("SELECT id FROM PersonData WHERE image_path = ?", (image_path,))
                    id_result = db_cursor.fetchone()
                    if id_result:
                        known_ids.append(id_result[0])
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
    
    return known_faces, known_ids

# Function to match the face from the webcam with the database
def match_face_from_db():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    print("Press 'q' to exit the program.")
    
    # Load known faces and ids from the database
    known_faces, known_ids = load_known_faces()

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        # Convert the frame to RGB (face_recognition format)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Compare the detected face with known faces from the database
            matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.6)
            behavior_score = None
            matched_id = None

            if True in matches:
                # Get the index of the matched face
                match_index = np.argmax(matches)
                matched_id = known_ids[match_index]

                # Fetch the behavior score of the matched person from the database
                db_cursor.execute("SELECT behavior_score FROM PersonData WHERE id = ?", (matched_id,))
                result = db_cursor.fetchone()
                if result:
                    behavior_score = result[0]
                    print(f"ID: {matched_id}, Behavior score: {behavior_score}")

                    # If behavior score is less than 35, show an alert
                    if behavior_score < 35:
                        cv2.putText(frame, "ALERT: Low Behavior Score", 
                                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    print(f"No behavior score found for ID {matched_id}")
            
            # Draw rectangle around the face and display the ID and behavior score
            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {matched_id} - Score: {behavior_score if behavior_score else 'N/A'}", 
                        (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Display the frame with detected faces
        cv2.imshow("Face Recognition - Match with Database", frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    db_connection.close()

# Run the face matching function
match_face_from_db()
