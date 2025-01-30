import numpy as np
import cv2
import mediapipe as mp

class Test:
    def __init__(self):
        try:
            data = np.load('model_weights.npz')
            self.hidden_layer_1_w = data['hidden_layer_1_w']
            self.hidden_layer_1_b = data['hidden_layer_1_b']
            self.output_layer_w = data['output_layer_w']
            self.output_layer_b = data['output_layer_b']
        except FileNotFoundError:
            print("Error: model_weights.npz not found.")
            exit(1)

    def get_y_hat(self, landmarks):
        a1 = np.maximum(0, np.dot(landmarks, self.hidden_layer_1_w) + self.hidden_layer_1_b) 
        z2 = np.dot(a1, self.output_layer_w) + self.output_layer_b
        z2 -= np.max(z2, axis=1, keepdims=True)  
        exp_values = np.exp(z2)
        self.y_hat = exp_values / np.sum(exp_values, axis=1, keepdims=True)

mp_hand_module = mp.solutions.hands
hands = mp_hand_module.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.6)
mp_draw = mp.solutions.drawing_utils
vid = cv2.VideoCapture(0)

while True:
    success, frame = vid.read()
    if not success:
        print("Failed to read frame.")
        break

    frame = cv2.flip(frame, 1) 
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb_frame)
    if result.multi_hand_landmarks:
        for handLm in result.multi_hand_landmarks:
            landmarks = []
            for lm in handLm.landmark:
                landmarks.append([lm.x, lm.y, lm.z])
            mp_draw.draw_landmarks(frame, handLm, mp_hand_module.HAND_CONNECTIONS)

        landmarks = np.array(landmarks)
        initial_points = landmarks[0] 

        landmarks -= initial_points

        flattened = landmarks.ravel()

        test_gesture = Test()
        test_gesture.get_y_hat(flattened.reshape(1, -1))

        predicted_class = np.argmax(test_gesture.y_hat)
  
        if predicted_class == 0:
            gesture_label = "open"
        elif predicted_class == 1:
            gesture_label = "close"
        elif predicted_class == 2:
            gesture_label = "fcuk_off"
        elif predicted_class == 3:
            gesture_label = "thumbs_up"

        cv2.putText(frame, f"Gesture: {gesture_label}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
