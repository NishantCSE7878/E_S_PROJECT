import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import datetime

# --- CONFIGURATION ---
START_TIME = datetime.time(18, 0, 0) # 6:00 PM
END_TIME = datetime.time(22, 0, 0)   # 10:00 PM

class SignLanguageApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Sign Language Detection System")
        self.window.geometry("900x700")

        # Check Time Constraint immediately
        if not self.is_within_time():
            messagebox.showerror("Access Denied", f"System is only operational between 6 PM and 10 PM.\nCurrent time: {datetime.datetime.now().strftime('%H:%M')}")
            self.window.destroy()
            return

        # Load trained model and classes
        try:
            self.model = tf.keras.models.load_model('sign_model.h5')
            with open('classes.txt', 'r') as f:
                self.class_names = f.read().splitlines()
        except Exception as e:
            messagebox.showerror("Error", "Could not load model! Did you run train_model.py first?")
            self.window.destroy()
            return

        # GUI Elements
        tk.Label(window, text="Sign Language Detector", font=("Arial", 20, "bold")).pack(pady=10)
        
        btn_frame = tk.Frame(window)
        btn_frame.pack(pady=10)
        
        tk.Button(btn_frame, text="Upload Image", command=self.process_image, width=15, bg="lightblue").grid(row=0, column=0, padx=10)
        tk.Button(btn_frame, text="Upload Video", command=self.process_video, width=15, bg="lightgreen").grid(row=0, column=1, padx=10)

        self.canvas = tk.Canvas(window, width=640, height=480, bg="black")
        self.canvas.pack(pady=10)
        
        self.result_label = tk.Label(window, text="Prediction: --", font=("Arial", 18, "bold"), fg="blue")
        self.result_label.pack(pady=10)
        
        self.is_video_playing = False

    def is_within_time(self):
        """Check if current time is between 6 PM and 10 PM"""
        now = datetime.datetime.now().time()
        return START_TIME <= now <= END_TIME

    def predict_frame(self, frame):
        """Resize frame, pass to model, return prediction string"""
        # Resize to match training data (64x64)
        img = cv2.resize(frame, (64, 64))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_array = tf.expand_dims(img, 0) # Create a batch
        
        predictions = self.model.predict(img_array, verbose=0)
        score = tf.nn.softmax(predictions[0])
        predicted_class = self.class_names[np.argmax(score)]
        confidence = 100 * np.max(score)
        
        return f"{predicted_class} ({confidence:.1f}%)"

    def process_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png")])
        if not file_path: return
        self.is_video_playing = False

        frame = cv2.imread(file_path)
        prediction = self.predict_frame(frame)
        
        cv2.putText(frame, prediction, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        self.display_image(frame)
        self.result_label.config(text=f"Prediction: {prediction}")

    def process_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi")])
        if not file_path: return

        self.is_video_playing = True
        cap = cv2.VideoCapture(file_path)

        def stream():
            if not self.is_video_playing:
                cap.release()
                return

            ret, frame = cap.read()
            if not ret:
                cap.release()
                return

            prediction = self.predict_frame(frame)
            cv2.putText(frame, prediction, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            self.display_image(frame)
            self.result_label.config(text=f"Prediction: {prediction}")
            
            self.window.after(30, stream)

        stream()

    def display_image(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (640, 480))
        img_tk = ImageTk.PhotoImage(image=Image.fromarray(frame))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas.image = img_tk

if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.mainloop()