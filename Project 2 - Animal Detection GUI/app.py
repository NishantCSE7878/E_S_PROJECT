import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
import os

# --- CONFIGURATION ---
CARNIVORES = ['cat', 'dog', 'bear', 'lion', 'tiger', 'leopard', 'cheetah', 'wolf', 'fox', 'hyena'] 

class CustomAnimalDetectorApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Custom Animal Detection System")
        self.window.geometry("900x700")

        
        
        model_path = 'best.pt' 
        
        
        if os.path.exists(model_path):
            self.model = YOLO(model_path)
            print(f"Loaded CUSTOM model from {model_path}")
        else:
            self.model = YOLO('yolov8n.pt')
            print("Custom model not found. Loaded default yolov8n.pt")

        # GUI Elements
        self.label_title = tk.Label(window, text="Custom Animal Detection & Classification", font=("Arial", 20, "bold"))
        self.label_title.pack(pady=10)

        self.btn_frame = tk.Frame(window)
        self.btn_frame.pack(pady=10)

        self.btn_image = tk.Button(self.btn_frame, text="Upload Image", command=self.process_image, width=20, bg="lightblue")
        self.btn_image.grid(row=0, column=0, padx=10)

        self.btn_video = tk.Button(self.btn_frame, text="Upload Video", command=self.process_video, width=20, bg="lightgreen")
        self.btn_video.grid(row=0, column=1, padx=10)

        self.canvas = tk.Canvas(window, width=800, height=500, bg="black")
        self.canvas.pack(pady=10)

        self.is_video_playing = False

    def process_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if not file_path:
            return

        self.is_video_playing = False
        results = self.model(file_path)
        frame = results[0].plot(labels=False, boxes=False) # Get raw frame
        
        carnivore_count = 0
        
        # Draw boxes and apply logic
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            label = self.model.names[class_id].lower() # Get name and make lowercase
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = (0, 0, 255) if label in CARNIVORES else (0, 255, 0) # Red if Carnivore, else Green
            
            if label in CARNIVORES:
                carnivore_count += 1
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, f"{label.upper()}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        self.display_image(frame)
        
        if carnivore_count > 0:
            messagebox.showinfo("Detection Alert", f"Detected {carnivore_count} Carnivorous Animal(s)!")

    def process_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi")])
        if not file_path:
            return

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

            results = self.model(frame, stream=True)

            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    label = self.model.names[class_id].lower()
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    color = (0, 0, 255) if label in CARNIVORES else (0, 255, 0)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label.upper(), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            self.display_image(frame)
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
    app = CustomAnimalDetectorApp(root)
    root.mainloop()