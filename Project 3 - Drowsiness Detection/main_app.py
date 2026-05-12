import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from deepface import DeepFace
from tensorflow.keras.models import load_model

class MLDrowsinessDetector:
    def __init__(self, window):
        self.window = window
        self.window.title("ML Drowsiness & Age Detection System")
        self.window.geometry("1000x800")

        # Load OpenCV Cascades for localization
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
        
        # Load our trained Machine Learning Model
        try:
            self.model = load_model('eye_state_model.h5')
            print("ML Model loaded successfully.")
        except Exception as e:
            print("Error loading model. Did you run train_model.py first?")
            self.model = None

        # GUI Setup
        self.label_title = tk.Label(window, text="ML Drowsiness & Age Detector", font=("Arial", 24, "bold"))
        self.label_title.pack(pady=10)

        self.btn_frame = tk.Frame(window)
        self.btn_frame.pack(pady=10)

        self.btn_image = tk.Button(self.btn_frame, text="Upload Image", command=self.process_image, width=20, bg="lightblue")
        self.btn_image.grid(row=0, column=0, padx=10)

        self.btn_video = tk.Button(self.btn_frame, text="Upload Video", command=self.process_video, width=20, bg="lightgreen")
        self.btn_video.grid(row=0, column=1, padx=10)

        self.canvas = tk.Canvas(window, width=800, height=500, bg="black")
        self.canvas.pack(pady=10)

        self.is_running = False

    def analyze_age(self, face_img):
        try:
            analysis = DeepFace.analyze(face_img, actions=['age'], enforce_detection=False, silent=True)
            return int(analysis[0]['age'])
        except:
            return "N/A"

    def predict_eye_state(self, eye_img):
            """Passes the cropped eye to our trained CNN model"""
            if self.model is None: return 0 # Default to awake if model missing
            
            # FIX: Convert OpenCV's BGR format to RGB so it matches our training data!
            eye_img_rgb = cv2.cvtColor(eye_img, cv2.COLOR_BGR2RGB)
            
            # Preprocess image for the model
            eye_img_resized = cv2.resize(eye_img_rgb, (64, 64))
            eye_img_normalized = eye_img_resized / 255.0 # Normalize
            eye_img_batch = np.expand_dims(eye_img_normalized, axis=0) # Add batch dimension
            
            prediction = self.model.predict(eye_img_batch, verbose=0)
            raw_score = prediction[0][0]
            
            # DEBUG: Print to the terminal so we can see the math!
            # awake = 0, sleepy = 1
            print(f"Eye Score: {raw_score:.4f} | (Close to 0 = Awake, Close to 1 = Sleepy)")
            
            # If the score is closer to 0, return 0 (Awake). Otherwise return 1 (Sleepy).
            return 0 if raw_score < 0.5 else 1
    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        sleep_count = 0
        people_ages = []
        
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            # Age Prediction
            age = self.analyze_age(roi_color)
            people_ages.append(age)

            # Detect Eyes within the face
            eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 10)
            
            is_sleeping = True # Assume sleeping until proven awake
            
            for (ex, ey, ew, eh) in eyes:
                eye_roi = roi_color[ey:ey+eh, ex:ex+ew]
                state = self.predict_eye_state(eye_roi)
                
                if state == 0: # Model says at least one eye is Open
                    is_sleeping = False
                    break 

            # Drawing and Logic
            color = (0, 255, 0) # Green
            status = "Awake"
            
            if is_sleeping:
                status = "SLEEPING"
                color = (0, 0, 255) # Red
                sleep_count += 1

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
            cv2.putText(frame, f"{status} | Age: {age}", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return frame, len(faces), sleep_count, people_ages

    def process_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        if not file_path: return

        frame = cv2.imread(file_path)
        processed_frame, total, sleepers, ages = self.process_frame(frame)
        
        self.display_image(processed_frame)
        
        msg = f"Total People: {total}\nSleeping: {sleepers}\nAges: {ages}"
        messagebox.showinfo("Report", msg)

    def process_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi")])
        if not file_path: return

        self.is_running = True
        cap = cv2.VideoCapture(file_path)

        def stream():
            if not self.is_running:
                cap.release()
                return

            ret, frame = cap.read()
            if not ret:
                cap.release()
                return

            processed_frame, _, _, _ = self.process_frame(frame)
            self.display_image(processed_frame)
            self.window.after(10, stream)

        stream()

    def display_image(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (800, 500))
        img = Image.fromarray(frame)
        img_tk = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas.image = img_tk

if __name__ == "__main__":
    root = tk.Tk()
    app = MLDrowsinessDetector(root)
    root.mainloop()