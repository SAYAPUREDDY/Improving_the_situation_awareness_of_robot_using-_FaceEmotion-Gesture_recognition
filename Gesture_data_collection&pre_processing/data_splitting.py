import os
import shutil
from sklearn.model_selection import train_test_split

data_dir = "data"  
gestures = ["Hello", "Hold", "peace"] 
train_ratio = 0.8 

train_dir = os.path.join(data_dir, "Train")
test_dir = os.path.join(data_dir, "Test")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

for gesture in gestures:
    gesture_dir = os.path.join(data_dir, gesture)
    images = os.listdir(gesture_dir)
    train_images, test_images = train_test_split(images, train_size=train_ratio, random_state=42)

    
    train_gesture_dir = os.path.join(train_dir, gesture)
    test_gesture_dir = os.path.join(test_dir, gesture)
    os.makedirs(train_gesture_dir, exist_ok=True)
    os.makedirs(test_gesture_dir, exist_ok=True)


    for img_name in train_images:
        src_path = os.path.join(gesture_dir, img_name)
        dst_path = os.path.join(train_gesture_dir, img_name)
        shutil.move(src_path, dst_path)

    
    for img_name in test_images:
        src_path = os.path.join(gesture_dir, img_name)
        dst_path = os.path.join(test_gesture_dir, img_name)
        shutil.move(src_path, dst_path)

    print(f"Gesture: {gesture}")
    print(f"  Total images: {len(images)}")
    print(f"  Training images: {len(train_images)}")
    print(f"  Testing images: {len(test_images)}")

print("Data split complete.")
