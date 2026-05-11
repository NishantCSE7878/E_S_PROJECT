import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from deepface import DeepFace
import threading

class DrowsinessDetector:
    def __init__(self, window):
        self.window = window
        self.window.title("Drowsiness & Age Detection System")
        self.window.geometry("1000x800")

        # Load OpenCV Cascades
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

        # GUI Setup
        self.label_title = tk.Label(window, text="Drowsiness & Age Detector", font=("Arial", 24, "bold"))
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

    def analyze_face(self, face_img):
        """Helper to get age using DeepFace"""
        try:
            analysis = DeepFace.analyze(face_img, actions=['age'], enforce_detection=False, silent=True)
            return int(analysis[0]['age'])
        except:
            return "N/A"

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        sleep_count = 0
        people_ages = []
        
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            # Detect Eyes
            eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 10)
            
            # Age Prediction (Can be slow in real-time, so we do it per detection)
            age = self.analyze_face(roi_color)
            people_ages.append(age)

            # Drowsiness Logic: If no eyes detected in a face, mark as Sleeping
            status = "Awake"
            color = (0, 255, 0) # Green
            
            if len(eyes) == 0:
                status = "SLEEPING"
                color = (0, 0, 255) # Red
                sleep_count += 1

            # Draw Box and Labels
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

            # Note: DeepFace Age analysis on every frame of a video is heavy.
            # In a real app, we would only analyze age every 30 frames.
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
    app = DrowsinessDetector(root)
    root.mainloop()