import cv2
import numpy as np
import mediapipe as mp
from keras.models import model_from_json


gesture_dict = {0: "Hello", 1: "Hold", 2: "peace"}


json_file = open('gesture_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
gesture_model = model_from_json(loaded_model_json)
gesture_model.load_weights("gesture_model.h5")
print("Loaded gesture model from disk")


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    if not ret:
        break

   
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    result = hands.process(gray_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
  
            h, w, c = frame.shape
            x_min, x_max = w, 0
            y_min, y_max = h, 0
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min = min(x_min, x)
                x_max = max(x_max, x)
                y_min = min(y_min, y)
                y_max = max(y_max, y)
            
           
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            
            roi = frame[y_min:y_max, x_min:x_max]
            if roi.size == 0:
                continue
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            cropped_img_hand = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            
            
            gesture_prediction = gesture_model.predict(cropped_img_hand)
            maxindex_gesture = int(np.argmax(gesture_prediction))
            cv2.putText(frame, gesture_dict[maxindex_gesture], (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

  
    cv2.imshow('Gesture Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
