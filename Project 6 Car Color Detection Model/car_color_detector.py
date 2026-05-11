import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO

class TrafficAnalyzerApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Traffic & Car Color Analyzer")
        self.window.geometry("1000x750")

        # Load YOLOv8 Model (Automatically downloads yolov8n.pt if missing)
        print("Loading YOLO model...")
        self.model = YOLO('yolov8n.pt') 

        # GUI Setup
        self.title_label = tk.Label(window, text="Traffic Analysis System", font=("Arial", 22, "bold"))
        self.title_label.pack(pady=10)

        self.btn_upload = tk.Button(window, text="Upload Traffic Image", command=self.process_image, 
                                   width=25, height=2, bg="lightblue", font=("Arial", 12))
        self.btn_upload.pack(pady=10)

        # Stats Frame
        self.stats_frame = tk.Frame(window)
        self.stats_frame.pack(pady=5)
        
        self.car_lbl = tk.Label(self.stats_frame, text="Cars: 0", font=("Arial", 16, "bold"), fg="blue")
        self.car_lbl.grid(row=0, column=0, padx=20)
        
        self.person_lbl = tk.Label(self.stats_frame, text="People: 0", font=("Arial", 16, "bold"), fg="green")
        self.person_lbl.grid(row=0, column=1, padx=20)

        self.canvas = tk.Canvas(window, width=800, height=500, bg="black")
        self.canvas.pack(pady=10)

    def is_car_blue(self, car_roi):
        """Checks if the dominant color in the cropped car image is Blue"""
        # Convert BGR to HSV
        hsv = cv2.cvtColor(car_roi, cv2.COLOR_BGR2HSV)
        
        # Define range for Blue color in HSV
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
        
        # Create a mask that isolates blue pixels
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Calculate the percentage of blue pixels in the car region
        blue_pixels = cv2.countNonZero(mask)
        total_pixels = car_roi.shape[0] * car_roi.shape[1]
        
        if total_pixels == 0:
            return False
            
        blue_ratio = blue_pixels / total_pixels
        
        # If more than 15% of the car's bounding box is blue, we consider it a blue car
        return blue_ratio > 0.15

    def process_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        if not file_path: return

        frame = cv2.imread(file_path)
        
        # Run YOLOv8 inference
        results = self.model(frame)
        
        car_count = 0
        person_count = 0

        for box in results[0].boxes:
            class_id = int(box.cls[0])
            label = self.model.names[class_id]
            
            # Get coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            if label == 'person':
                person_count += 1
                # Draw a simple green box for people
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Person", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
            elif label in ['car', 'truck', 'bus']: # Catch all vehicle types just in case
                car_count += 1
                
                # Crop the car from the image to analyze its color
                car_roi = frame[y1:y2, x1:x2]
                
                # Project Logic: Red rectangle for Blue cars, Blue rectangle for other cars
                # OpenCV uses BGR format, so Red is (0,0,255) and Blue is (255,0,0)
                if self.is_car_blue(car_roi):
                    color = (0, 0, 255) # RED
                    text = "Blue Car"
                else:
                    color = (255, 0, 0) # BLUE
                    text = "Car (Other)"
                    
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Update GUI Labels
        self.car_lbl.config(text=f"Cars Detected: {car_count}")
        self.person_lbl.config(text=f"People Detected: {person_count}")

        self.display_image(frame)

    def display_image(self, frame):
        # Convert BGR (OpenCV) to RGB (Tkinter/Pillow)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize for preview while keeping aspect ratio roughly intact
        frame = cv2.resize(frame, (800, 500))
        img_tk = ImageTk.PhotoImage(image=Image.fromarray(frame))
        
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas.image = img_tk

if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficAnalyzerApp(root)
    root.mainloop()