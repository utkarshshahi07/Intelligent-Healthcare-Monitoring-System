import cv2
import face_recognition
import numpy as np
import sqlite3
from keras.models import load_model
from datetime import datetime, timedelta
import mediapipe as mp

# Load pre-trained face detection model
face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

# Load your emotion detection model
try:
    emotion_model = load_model('./model.h5')  # Ensure the model is in the same directory
    print("Emotion detection model loaded successfully.")
except Exception as e:
    print(f"Error loading emotion detection model: {e}")
    exit()

# Emotion class labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Angry', 'Surprise']

# Connect to the database
db_name = 'face_recognition.db'
conn = sqlite3.connect(db_name)
cursor = conn.cursor()

# Load known faces and encodings from the database
known_face_encodings = []
known_face_ids = []

cursor.execute("SELECT id, image_path FROM PersonData")
for row in cursor.fetchall():
    person_id, image_path = row
    try:
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(encoding)
        known_face_ids.append(person_id)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# To track angry emotion and gesture duration
angry_timestamps = {}
gesture_timestamps = {}

# Functions for Gesture Detection
def detect_fist(landmarks):
    distances = [
        np.linalg.norm([landmarks[8].x - landmarks[0].x, landmarks[8].y - landmarks[0].y]),
        np.linalg.norm([landmarks[12].x - landmarks[0].x, landmarks[12].y - landmarks[0].y]),
        np.linalg.norm([landmarks[16].x - landmarks[0].x, landmarks[16].y - landmarks[0].y]),
        np.linalg.norm([landmarks[20].x - landmarks[0].x, landmarks[20].y - landmarks[0].y]),
    ]
    return all(dist < 0.1 for dist in distances)

def detect_aggressive_pointing(landmarks):
    index_extended = landmarks[8].y < landmarks[6].y
    others_curled = all(landmarks[finger_tip].y > landmarks[finger_base].y for finger_tip, finger_base in [(12, 10), (16, 14), (20, 18)])
    return index_extended and others_curled

# Open the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    # Flip and convert frame
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Get face locations and encodings
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Process hand gestures
    results = hands.process(rgb_frame)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Match face with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances) if matches else None

        if best_match_index is not None and matches[best_match_index]:
            # Face matched, get person_id
            person_id = known_face_ids[best_match_index]

            # Get behavior score from the database
            cursor.execute("SELECT behavior_score FROM PersonData WHERE id = ?", (person_id,))
            row = cursor.fetchone()
            if row:
                behavior_score = row[0]

                # Get face bounding box
                top, right, bottom, left = face_location
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                # Crop face for emotion detection
                roi_gray = gray_frame[top:bottom, left:right]
                if roi_gray.size > 0:  # Ensure the region is valid
                    roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
                    roi_gray = roi_gray.astype('float32') / 255.0
                    roi_gray = np.expand_dims(roi_gray, axis=-1)
                    roi_gray = np.expand_dims(roi_gray, axis=0)

                    # Predict emotion
                    preds = emotion_model.predict(roi_gray, verbose=0)[0]
                    emotion = emotion_labels[np.argmax(preds)]

                    # Display emotion on screen
                    cv2.putText(
                        frame,
                        f"ID: {person_id}, Emotion: {emotion}, Score: {behavior_score}",
                        (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )

                    # Check if emotion is angry
                    current_time = datetime.now()
                    if emotion == 'Angry':
                        if person_id in angry_timestamps:
                            elapsed_time = (current_time - angry_timestamps[person_id]).total_seconds()
                            if elapsed_time >= 5:  # Angry for more than 5 seconds
                                behavior_score = max(behavior_score - 10, 0)
                                cursor.execute("UPDATE PersonData SET behavior_score = ? WHERE id = ?", (behavior_score, person_id))
                                conn.commit()
                                print(f"Behavior score updated: ID {person_id}, New Score: {behavior_score}")
                                angry_timestamps.pop(person_id)  # Reset timer
                        else:
                            angry_timestamps[person_id] = current_time

    # Detect hand gestures
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Gesture detection
            if detect_fist(hand_landmarks.landmark):
                cv2.putText(frame, 'Fist Detected', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                current_time = datetime.now()
                if person_id in gesture_timestamps:
                    elapsed_time = (current_time - gesture_timestamps[person_id]).total_seconds()
                    if elapsed_time >= 3:
                        behavior_score = max(behavior_score - 10, 0)
                        cursor.execute("UPDATE PersonData SET behavior_score = ? WHERE id = ?", (behavior_score, person_id))
                        conn.commit()
                        print(f"Behavior score updated (Gesture): ID {person_id}, New Score: {behavior_score}")
                        gesture_timestamps.pop(person_id)
                else:
                    gesture_timestamps[person_id] = current_time

            elif detect_aggressive_pointing(hand_landmarks.landmark):
                cv2.putText(frame, 'Aggressive Pointing Detected', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                current_time = datetime.now()
                if person_id in gesture_timestamps:
                    elapsed_time = (current_time - gesture_timestamps[person_id]).total_seconds()
                    if elapsed_time >= 3:
                        behavior_score = max(behavior_score - 10, 0)
                        cursor.execute("UPDATE PersonData SET behavior_score = ? WHERE id = ?", (behavior_score, person_id))
                        conn.commit()
                        print(f"Behavior score updated (Gesture): ID {person_id}, New Score: {behavior_score}")
                        gesture_timestamps.pop(person_id)
                else:
                    gesture_timestamps[person_id] = current_time

    # Show the video feed
    cv2.imshow("Face & Gesture Recognition", frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
conn.close()
cv2.destroyAllWindows()