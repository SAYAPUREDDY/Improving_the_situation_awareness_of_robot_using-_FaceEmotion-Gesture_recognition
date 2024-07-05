import cv2
import numpy as np
import time
import os

hand_cascade = cv2.CascadeClassifier('hand.xml')

offset = 20
imgSize = 48
folder = "data/Hello"  
counter = 0

if not os.path.exists(folder):
    os.makedirs(folder)

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hands = hand_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in hands:
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
    if key == ord("s") and len(hands) > 0:
        counter += 1
        save_path = f'{folder}/Image_{time.time()}.jpg'
        cv2.imwrite(save_path, imgResize)
        print(f"Saved image {counter} at {save_path}")

cap.release()
cv2.destroyAllWindows()
