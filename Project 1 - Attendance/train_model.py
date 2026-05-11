import os
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from deepface import DeepFace

# Configuration
DATASET_PATH = "dataset"
MODEL_SAVE_PATH = "trained_model.pkl"

embeddings = []
labels = []

print("🔄 Training started. Extracting features from photos...")

# Loop through each student folder
for student_name in os.listdir(DATASET_PATH):
    student_folder = os.path.join(DATASET_PATH, student_name)
    
    if os.path.isdir(student_folder):
        for img_name in os.listdir(student_folder):
            img_path = os.path.join(student_folder, img_name)
            try:
                # Extract the face "fingerprint" (embedding)
                result = DeepFace.represent(img_path=img_path, model_name="VGG-Face", enforce_detection=True)
                embedding = result[0]["embedding"]
                
                embeddings.append(embedding)
                labels.append(student_name)
                print(f"✅ Processed: {student_name} ({img_name})")
            except Exception as e:
                print(f"❌ Skipping {img_name}: No face detected.")

# Convert labels to numbers (e.g., "John" -> 0)
le = LabelEncoder()
encoded_labels = le.fit_transform(labels)

# TRAIN THE SVM MODEL
print(f"🧠 Training SVM model on {len(embeddings)} images...")
classifier = SVC(kernel='linear', probability=True)
classifier.fit(embeddings, encoded_labels)

# Save the model and label encoder to a file
with open(MODEL_SAVE_PATH, 'wb') as f:
    pickle.dump((classifier, le), f)

print(f"\n🎉 Success! Custom model saved as '{MODEL_SAVE_PATH}'")