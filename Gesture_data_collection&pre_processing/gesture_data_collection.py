import cv2
import mediapipe as mp
import numpy as np
import time
import os


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


offset = 20
imgSize = 48
folder = "data/peace"  
counter = 0


if not os.path.exists(folder):
    os.makedirs(folder)

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
         
            h, w, c = img.shape
            x_min, x_max, y_min, y_max = w, 0, h, 0

            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x < x_min: x_min = x
                if x > x_max: x_max = x
                if y < y_min: y_min = y
                if y > y_max: y_max = y

            x, y, w, h = x_min, y_min, x_max - x_min, y_max - y_min

            
            x1, y1 = max(0, x - offset), max(0, y - offset)
            x2, y2 = min(img.shape[1], x + w + offset), min(img.shape[0], y + h + offset)

            imgCrop = img[y1:y2, x1:x2]

            
            imgCropShape = imgCrop.shape
            if imgCropShape[0] > 0 and imgCropShape[1] > 0:
                
                imgResize = cv2.resize(imgCrop, (imgSize, imgSize))
                
                
                cv2.imshow("ImageCrop", imgCrop)
                cv2.imshow("ImageResized", imgResize)

    
    cv2.imshow("Image", img)

    
    key = cv2.waitKey(1)
    if key == ord("s") and results.multi_hand_landmarks:
        counter += 1
        save_path = f'{folder}/Image_{time.time()}.jpg'
        cv2.imwrite(save_path, imgResize)
        print(f"Saved image {counter} at {save_path}")

cap.release()
cv2.destroyAllWindows()


