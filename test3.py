import cv2
import mediapipe as mp
import numpy as np
import sqlite3
from keras.models import load_model
from datetime import datetime

# Initialize MediaPipe Hands and Drawing Utils
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5
)

# Load pre-trained classifier (Haar Cascade for face detection)
face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

# Load your trained emotion detection model
try:
    emotion_model = load_model('./model.h5')  # Load your trained emotion detection model
    print("Emotion detection model loaded successfully.")
except Exception as e:
    print(f"Error loading emotion detection model: {e}")
    exit()

# Emotion class labels (ensure these match the model output)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Angry', 'Surprise']

# SQLite Database Setup
db_name = 'incident_logs.db'
conn = sqlite3.connect(db_name)
cursor = conn.cursor()

# Check if 'type' column exists and add it if missing
cursor.execute("PRAGMA table_info(incidents);")
columns = [column[1] for column in cursor.fetchall()]
if 'type' not in columns:
    cursor.execute('''ALTER TABLE incidents ADD COLUMN type TEXT;''')
    conn.commit()

# Create table if it doesn't already exist
cursor.execute(''' 
CREATE TABLE IF NOT EXISTS incidents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    type TEXT,
    label TEXT,
    description TEXT
)
''')
conn.commit()

# Define Restricted Area (x1, y1, x2, y2)
RESTRICTED_AREA = (100, 100, 250, 250)  # Adjust as needed

# Function to calculate the Euclidean distance between two points
def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# Functions for Gesture Detection
def detect_fist(landmarks):
    # Check if all fingers are curled (fist gesture)
    distances = [
        np.linalg.norm([landmarks[8].x - landmarks[0].x, landmarks[8].y - landmarks[0].y]),
        np.linalg.norm([landmarks[12].x - landmarks[0].x, landmarks[12].y - landmarks[0].y]),
        np.linalg.norm([landmarks[16].x - landmarks[0].x, landmarks[16].y - landmarks[0].y]),
        np.linalg.norm([landmarks[20].x - landmarks[0].x, landmarks[20].y - landmarks[0].y]),
    ]
    return all(dist < 0.1 for dist in distances)

def detect_aggressive_pointing(landmarks):
    # Index finger extended, others curled
    index_extended = landmarks[8].y < landmarks[6].y
    others_curled = all(landmarks[finger_tip].y > landmarks[finger_base].y for finger_tip, finger_base in [(12, 10), (16, 14), (20, 18)])
    return index_extended and others_curled

# Add Optical Flow Detection for Vigorous Motion
def detect_vigorous_motion(prev_gray, curr_gray, threshold=5000):
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    if prev_pts is not None:
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
        if next_pts is not None:
            flow = next_pts - prev_pts
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            total_magnitude = np.sum(magnitude)
            if total_magnitude > threshold:
                return True
    return False

# Define proximity threshold
PROXIMITY_THRESHOLD = 100  # Adjust based on camera resolution

# Open webcam for real-time detection
cap = cv2.VideoCapture(0)

# Initialize previous frame for optical flow calculation
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    # Flip and preprocess the frame
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_height, frame_width, _ = frame.shape

    # Detect faces for emotion recognition
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    face_centers = []

    for (x, y, w, h) in faces:
        # Calculate the center of the face
        face_center = (x + w // 2, y + h // 2)
        face_centers.append(face_center)

        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Crop face and resize for emotion detection
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        roi_gray = roi_gray.astype('float32') / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=-1)
        roi_gray = np.expand_dims(roi_gray, axis=0)

        # Predict emotion
        if np.sum([roi_gray]) != 0:
            preds = emotion_model.predict(roi_gray)[0]
            emotion = emotion_labels[preds.argmax()]
            label_position = (x, y - 10)
            cv2.putText(frame, emotion, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Log emotion incidents
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            description = f"Detected emotion: {emotion}."
            cursor.execute('INSERT INTO incidents (timestamp, type, label, description) VALUES (?, ?, ?, ?)', 
                           (timestamp, 'Emotion', emotion, description))
            conn.commit()

            # Alert for angry emotion
            if emotion == 'Angry':
                print("Alert: Angry person detected!")

    # Check for close proximity between detected faces
    for i in range(len(face_centers)):
        for j in range(i + 1, len(face_centers)):
            distance = calculate_distance(face_centers[i], face_centers[j])
            if distance < PROXIMITY_THRESHOLD:
                cv2.putText(frame, "ALERT: Close Proximity!", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                print("Alert: Two or more persons are in close proximity!")

    # Detect hand gestures using MediaPipe Hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Gesture detection
            if detect_fist(hand_landmarks.landmark):
                cv2.putText(frame, 'ALERT: Fist Clenching!', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif detect_aggressive_pointing(hand_landmarks.landmark):
                cv2.putText(frame, 'ALERT: Aggressive Pointing!', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Detect vigorous motion
    if detect_vigorous_motion(prev_gray, gray, threshold=500):
        print("Vigorous motion detected! Sending alert...")

    # Draw restricted area
    cv2.rectangle(frame, (RESTRICTED_AREA[0], RESTRICTED_AREA[1]), (RESTRICTED_AREA[2], RESTRICTED_AREA[3]), (0, 0, 255), 2)
    cv2.putText(frame, 'Restricted Area', (RESTRICTED_AREA[0], RESTRICTED_AREA[1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Emotion & Gesture Detection', frame)

    # Update the previous frame for the next iteration
    prev_gray = gray

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
conn.close()
cv2.destroyAllWindows()
