import cv2
import mediapipe as mp
import json

import time

# Mediapipe setup
mphandmodul = mp.solutions.hands
hands = mphandmodul.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.6)
mpdraw = mp.solutions.drawing_utils
captures = 0

# Video capture
vid = cv2.VideoCapture(0)
data_file = r"data.json"
gesture_data = []
prev_time = 0
interval = 0.03
temp = 0
try:
    with open(data_file, "r") as f:
        gesture_data = json.load(f)
        print(f"Loaded {len(gesture_data)} gestures from '{data_file}'.")
except (json.JSONDecodeError, FileNotFoundError, IOError) as e:
    print(f"Error loading data: {e}. Starting fresh.")
    gesture_data = []


print("constantly start moving your hand in the given gesture, the camera will auto capture the landmarks.")
print("Press 'q' to quit and save the data to a JSON file.")
landmarks = []
while True:
    success, frame = vid.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb_frame)
    
    if result.multi_hand_landmarks:
        for handLm in result.multi_hand_landmarks:
            landmarks = []
            for lm in handLm.landmark:
                landmarks.append([lm.x, lm.y, lm.z])  
            cv2.putText(frame, f"DATA_Recorded : {captures}/1200", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            mpdraw.draw_landmarks(frame, handLm, mphandmodul.HAND_CONNECTIONS)

    current_time = time.time()
    cv2.imshow("Hand Gesture Data Collection", frame)
    key = cv2.waitKey(1) & 0xFF
    if current_time - prev_time >= interval:  
        prev_time = current_time
        label = "thumbs_up"
        if landmarks != []:
            gesture_data.append({"gesture": label,"landmarks": landmarks})
     
        captures += 1
    elif key == ord('q'):
        break
    elif captures >= 1200:
        break

vid.release()
cv2.destroyAllWindows()

with open(r"data.json", "w") as f:
    json.dump(gesture_data, f, indent=4)
