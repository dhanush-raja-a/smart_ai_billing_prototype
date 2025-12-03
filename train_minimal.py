from ultralytics import YOLO

def main():
    # load a small pre-trained YOLOv8 model
    model = YOLO("yolov8n.pt")

    # train on your tiny 2-class dataset
    model.train(
        data="/Users/dhanushrajaa/Desktop/yolo model prototype/mouse/data.yaml",  # path to your yaml
        epochs=20,                          # small number, just to see results
        imgsz=640,
        batch=8
    )

if __name__ == "__main__":
    main()