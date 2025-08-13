from keras.models import load_model
import cv2
import numpy as np
import sqlite3
from datetime import datetime

# Load pre-trained classifier (Haar Cascade for face detection)
face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

# Load your trained emotion detection model
try:
    classifier = load_model('./model.h5')  # Load your trained emotion detection model
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Emotion class labels (ensure these match the model output)
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# SQLite Database Setup
db_name = 'incident_logs.db'
conn = sqlite3.connect(db_name)
cursor = conn.cursor()

# Create table if it doesn't already exist
cursor.execute('''
CREATE TABLE IF NOT EXISTS incidents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    label TEXT,
    description TEXT
)
''')
conn.commit()

# Define a restricted area (x1, y1, x2, y2)
RESTRICTED_AREA = (100, 100, 400, 400)  # Adjust as needed

# Distance threshold (in pixels) for alert when people come close
CLOSE_DISTANCE_THRESHOLD = 100  # You can adjust this value

# Open webcam for real-time emotion detection
cap = cv2.VideoCapture(0)

while True:
    # Grab a single frame of video
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for face detection

    # Detect faces in the image
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    current_positions = []  # Store current positions of faces

    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Crop the face region and resize it to 48x48
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        # Normalize and prepare input for the model
        roi_gray = roi_gray.astype('float32') / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=-1)  # Add channel dimension
        roi_gray = np.expand_dims(roi_gray, axis=0)  # Add batch dimension

        # Make a prediction on the face region
        if np.sum([roi_gray]) != 0:
            preds = classifier.predict(roi_gray)[0]
            label = class_labels[preds.argmax()]  # Get the label with the highest probability
            label_position = (x, y - 10)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Log the incident to the database
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            description = f"Detected emotion: {label}."
            cursor.execute('INSERT INTO incidents (timestamp, label, description) VALUES (?, ?, ?)', 
                           (timestamp, label, description))
            conn.commit()
            print(f"Logged to database: {label} at {timestamp}")

            # Example alert for "Angry" emotion
            if label == 'Angry':
                print("Alert: Angry person detected!")
        else:
            cv2.putText(frame, 'No Face Found', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Store the current person's position (center of the detected face)
        current_positions.append((x + w // 2, y + h // 2))

    # Draw restricted area on the frame
    cv2.rectangle(frame, (RESTRICTED_AREA[0], RESTRICTED_AREA[1]),
                  (RESTRICTED_AREA[2], RESTRICTED_AREA[3]), (0, 0, 255), 2)
    cv2.putText(frame, 'Restricted Area', (RESTRICTED_AREA[0], RESTRICTED_AREA[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Check if anyone is inside the restricted area
    for pos in current_positions:
        if RESTRICTED_AREA[0] <= pos[0] <= RESTRICTED_AREA[2] and RESTRICTED_AREA[1] <= pos[1] <= RESTRICTED_AREA[3]:
            cv2.putText(frame, 'Alert: Person in Restricted Area!', (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            print("Alert: A person is in the restricted area!")

    # Check if two or more people are too close to each other
    for i in range(len(current_positions)):
        for j in range(i + 1, len(current_positions)):
            x1, y1 = current_positions[i]
            x2, y2 = current_positions[j]
            distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            if distance < CLOSE_DISTANCE_THRESHOLD:
                cv2.putText(frame, 'Alert: People too close!', (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                print("Alert: Two people are too close to each other!")

    # Display the resulting frame
    cv2.imshow('Emotion & Incident Logger', frame)

    # Press 'q' to exit the video stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the database connection
cap.release()
conn.close()
cv2.destroyAllWindows()