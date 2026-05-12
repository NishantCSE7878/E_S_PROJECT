from ultralytics import YOLO

def main():
    # Load the base model 
    model = YOLO('yolov8n.pt') 

    # TRAIN THE MODEL

    print("Starting training process...")
    results = model.train(
        data='dataset\data.yaml', 
        epochs=20, 
        imgsz=640, 
        plots=True
    )
    
    print("Training complete! Your custom model is saved in: runs/detect/train/weights/best.pt")

if __name__ == '__main__':
    
    main()