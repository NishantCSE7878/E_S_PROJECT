import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

print("Starting Model Training...")

# 1. Prepare the Data
# Ensure you have a 'dataset/train' folder with 'Open_Eyes' and 'Closed_Eyes' subfolders
data_dir = 'drowsiness datasets\data\train' 

# Normalize pixel values to be between 0 and 1
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    data_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary', # 0 for Closed, 1 for Open
    subset='training'
)

val_data = datagen.flow_from_directory(
    data_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# 2. Build the CNN Architecture
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5), # Prevents overfitting
    Dense(1, activation='sigmoid') # Outputs a probability between 0 and 1
])

# 3. Compile and Train
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("Training started. This may take a few minutes...")
model.fit(train_data, validation_data=val_data, epochs=10)

# 4. Save the Model
model.save('eye_state_model.h5')
print("Model saved successfully as 'eye_state_model.h5'!")