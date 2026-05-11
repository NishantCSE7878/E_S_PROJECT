import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from deepface import DeepFace

class NationalityApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Nationality & Emotion Analyzer")
        self.window.geometry("1100x700")

        # GUI Components
        self.title = tk.Label(window, text="Nationality Detection System", font=("Arial", 22, "bold"))
        self.title.pack(pady=10)

        self.main_frame = tk.Frame(window)
        self.main_frame.pack(pady=10)

        # Left side: Image Preview
        self.canvas = tk.Canvas(self.main_frame, width=600, height=450, bg="gray")
        self.canvas.grid(row=0, column=0, padx=20)

        # Right side: Results Panel
        self.result_box = tk.Text(self.main_frame, width=40, height=20, font=("Arial", 12))
        self.result_box.grid(row=0, column=1, padx=20)

        self.btn_upload = tk.Button(window, text="Upload Image", command=self.process_image, 
                                   width=20, height=2, bg="orange", font=("Arial", 10, "bold"))
        self.btn_upload.pack(pady=20)

    def get_color_name(self, bgr_color):
        """Simple heuristic to name a color based on BGR values"""
        b, g, r = bgr_color
        if r > g and r > b: return "Red"
        if g > r and g > b: return "Green"
        if b > r and b > g: return "Blue"
        if r > 200 and g > 200 and b < 100: return "Yellow"
        if r < 50 and g < 50 and b < 50: return "Black"
        if r > 200 and g > 200 and b > 200: return "White"
        return "Mixed/Neutral"

    def process_image(self):
        file_path = filedialog.askopenfilename()
        if not file_path: return

        # Load and Display Image
        img = cv2.imread(file_path)
        display_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        display_img = cv2.resize(display_img, (600, 450))
        img_tk = ImageTk.PhotoImage(Image.fromarray(display_img))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas.image = img_tk

        self.result_box.delete('1.0', tk.END)
        self.result_box.insert(tk.END, "Analyzing... please wait.\n")
        self.window.update()

        try:
            # 1. AI Analysis
            objs = DeepFace.analyze(img_path=file_path, actions=['age', 'gender', 'race', 'emotion'], enforce_detection=False)
            res = objs[0]
            
            race = res['dominant_race']
            emotion = res['dominant_emotion']
            age = res['age']
            
            # 2. Coordinate logic for Dress Color
            region = res['region'] # x, y, w, h of face
            # Look at a small box below the chin for dress color
            dress_y = region['y'] + region['h'] + 20
            dress_x = region['x'] + (region['w'] // 2)
            
            # Ensure coordinates are within image bounds
            h_img, w_img, _ = img.shape
            dress_y = min(dress_y, h_img - 10)
            dress_x = min(dress_x, w_img - 10)
            
            sample_area = img[dress_y:dress_y+20, dress_x-10:dress_x+10]
            avg_color_bgr = np.mean(sample_area, axis=(0, 1))
            dress_color = self.get_color_name(avg_color_bgr)

            # 3. Project Requirements Logic
            output = f"--- RESULTS ---\n\n"
            
            if race == 'indian':
                output += f"Nationality: Indian 🇮🇳\n"
                output += f"Emotion: {emotion}\n"
                output += f"Predicted Age: {age}\n"
                output += f"Dress Color: {dress_color}\n"
            
            elif race in ['white', 'latino hispanic']:
                output += f"Nationality: United States 🇺🇸\n"
                output += f"Emotion: {emotion}\n"
                output += f"Predicted Age: {age}\n"
                
            elif race == 'black':
                output += f"Nationality: African 🌍\n"
                output += f"Emotion: {emotion}\n"
                output += f"Dress Color: {dress_color}\n"
                
            else:
                output += f"Nationality: {race.capitalize()}\n"
                output += f"Emotion: {emotion}\n"

            self.result_box.delete('1.0', tk.END)
            self.result_box.insert(tk.END, output)

        except Exception as e:
            self.result_box.insert(tk.END, f"\nError: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = NationalityApp(root)
    root.mainloop()