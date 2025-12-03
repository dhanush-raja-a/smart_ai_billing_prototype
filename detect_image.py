from ultralytics import YOLO

def main():
    model = YOLO("runs/detect/train/weights/best.pt")  # your trained model

    results = model("/Users/dhanushrajaa/Desktop/yolo model prototype/mouse/train/images/0a538666793eb2c9_jpg.rf.bd14c3f1723053f102ad2143b9064812.jpg", show=True, save=True)
    # Put a test.jpg with your apples/bananas in project root

if __name__ == "__main__":
    main()