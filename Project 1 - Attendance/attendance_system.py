import cv2
import pandas as pd
import datetime
import pickle
import os
from deepface import DeepFace

# --- CONFIGURATION ---
MODEL_PATH = "trained_model.pkl"
CSV_FILE = "attendance_log.csv"
START_TIME = datetime.time(9, 30, 0)   # 9:30 AM (Start of day)
END_TIME = datetime.time(10, 0, 0)  # 10:00 AM (End of day)

# Check if model exists
if not os.path.exists(MODEL_PATH):
    print("❌ Error: trained_model.pkl not found! Run train_model.py first.")
    exit()

# Load your custom trained model
with open(MODEL_PATH, 'rb') as f:
    clf, le = pickle.load(f)

attendance_list = []
marked_students = set()

def is_attendance_open():
    now = datetime.datetime.now().time()
    return START_TIME <= now <= END_TIME

# Start Video
cap = cv2.VideoCapture(0)
print("📸 System Online. Monitoring for students...")

while True:
    ret, frame = cap.read()
    if not ret: break

    current_time = datetime.datetime.now().time()
    
    if not is_attendance_open():
        status_text = "SYSTEM CLOSED (Open 09:30 - 10:00)"
        color = (0, 0, 255)
    else:
        status_text = "SYSTEM ACTIVE: Scanning..."
        color = (0, 255, 0)
        
        try:
            # 1. Get embedding from the live frame
            face_objs = DeepFace.represent(img_path=frame, model_name="VGG-Face", enforce_detection=False)
            
            if face_objs:
                for face in face_objs:
                    embedding = face["embedding"]
                    # 2. Use YOUR trained model to predict the student
                    prediction = clf.predict([embedding])
                    probability = clf.predict_proba([embedding]).max()
                    
                    # Only accept if confidence is above 60%
                    if probability > 0.40:
                        student_name = le.inverse_transform(prediction)[0]
                        
                        # 3. Detect Emotion
                        analysis = DeepFace.analyze(img_path=frame, actions=['emotion'], enforce_detection=False)
                        emotion = analysis[0]['dominant_emotion']
                        
                        # 4. Mark Attendance
                        if student_name not in marked_students:
                            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            attendance_list.append({
                                "Student": student_name, 
                                "Time": timestamp, 
                                "Emotion": emotion, 
                                "Status": "Present"
                            })
                            marked_students.add(student_name)
                            print(f"✅ Marked: {student_name} | Emotion: {emotion}")

                        # Draw result on screen
                        x, y, w, h = face["facial_area"]["x"], face["facial_area"]["y"], face["facial_area"]["w"], face["facial_area"]["h"]
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, f"{student_name} ({emotion})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        except Exception as e:
            pass # No face found in this frame

    # UI Overlay
    cv2.putText(frame, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.imshow("AI Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save to CSV
if attendance_list:
    df = pd.DataFrame(attendance_list)
    df.to_csv(CSV_FILE, index=False)
    print(f"📁 Attendance saved to {CSV_FILE}")
else:
    print("⚠️ No data recorded.")

cap.release()
cv2.destroyAllWindows()