import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Path to dataset (ensure you have the correct path)
DIR = r'E:/cvprr/Attendance System/dataset'
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def get_images_and_labels(path):
    # Get the path of all the files in the folder
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    Ids = []

    # Loop through all the image paths
    for imagePath in imagePaths:
        pilImage = cv2.imread(imagePath)
        gray = cv2.cvtColor(pilImage, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            resized_face = cv2.resize(face, (150, 150))  # Resize the face image to the fixed size for CNN input
            faceSamples.append(resized_face)
            Id = int(os.path.split(imagePath)[-1].split(".")[1])  # Assuming label is part of file name
            Ids.append(Id)

    return faceSamples, Ids


# Load images and labels
faces, Ids = get_images_and_labels(DIR)

# Normalize faces by scaling pixel values between 0 and 1
faces = np.array(faces, dtype="float32") / 255.0
faces = faces.reshape(faces.shape[0], 150, 150, 1)  # Add channel dimension

# Convert Ids to numpy array
Ids = np.array(Ids)

# Split dataset into training and validation sets (80% train, 20% validation)
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(faces, Ids, test_size=0.2, random_state=42)

# Building CNN model
model = Sequential()

# Add convolutional layers and max-pooling layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the output from the convolutional layers
model.add(Flatten())

# Fully connected layers
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Dropout to avoid overfitting
model.add(Dense(len(set(Ids)), activation='softmax'))  # Output layer: one neuron per label (person)

# Compile the model
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Save the model after training
model.save('face_recognition_model.h5')
print("Model successfully saved!")

