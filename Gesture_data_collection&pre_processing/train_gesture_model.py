
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator


train_data_gen = ImageDataGenerator(rescale=1./255)
validation_data_gen = ImageDataGenerator(rescale=1./255)


train_generator = train_data_gen.flow_from_directory(
        'data/train',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

validation_generator = validation_data_gen.flow_from_directory(
        'data/test',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

gesture_model = Sequential()

gesture_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
gesture_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
gesture_model.add(MaxPooling2D(pool_size=(2, 2)))
gesture_model.add(Dropout(0.25))

gesture_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
gesture_model.add(MaxPooling2D(pool_size=(2, 2)))
gesture_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
gesture_model.add(MaxPooling2D(pool_size=(2, 2)))
gesture_model.add(Dropout(0.25))

gesture_model.add(Flatten())
gesture_model.add(Dense(1024, activation='relu'))
gesture_model.add(Dense(3, activation='softmax'))

cv2.ocl.setUseOpenCL(False)

gesture_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001,beta_1=0.9, beta_2=0.999, epsilon=1e-07), metrics=['accuracy'])

gesture_model_info = gesture_model.fit_generator(
        train_generator,
        steps_per_epoch=10088// 64,
        epochs=4,
        validation_data=validation_generator,
        validation_steps=2524// 64
        )

model_json = gesture_model.to_json()
with open("gesture_model.json", "w") as json_file:
    json_file.write(model_json)

gesture_model.save_weights('gesture_model.h5')