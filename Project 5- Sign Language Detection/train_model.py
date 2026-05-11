import tensorflow as tf
from tensorflow.keras import layers, models
import os

# --- CONFIGURATION ---
DATASET_DIR = 'dataset' # Folder containing your sign subfolders
IMG_SIZE = (64, 64)     # Resize images to speed up training
BATCH_SIZE = 32
EPOCHS = 5              # Keep it low for a quick test

print("Loading dataset...")
# Load images from folders
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_dataset.class_names
print(f"Classes found: {class_names}")

# Save class names for the GUI to use later
with open('classes.txt', 'w') as f:
    f.write('\n'.join(class_names))

# Build a lightweight CNN model
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

print("Starting training... (This might take a few minutes)")
model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS)

# Save the trained brain
model.save('sign_model.h5')
print("✅ Model trained and saved as 'sign_model.h5'")