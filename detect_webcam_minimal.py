from ultralytics import YOLO
import cv2

def main():
    model = YOLO("runs/detect/train/weights/best.pt")
    cap = cv2.VideoCapture(0)  # 0 = default webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)

        # Get detections from first result
        boxes = results[0].boxes
        class_ids = boxes.cls.cpu().numpy().astype(int) if boxes is not None else []

        # Simple count of each class per frame
        counts = {}
        for cid in class_ids:
            name = model.names[cid]
            counts[name] = counts.get(name, 0) + 1

        # Draw results on frame
        annotated_frame = results[0].plot()

        # Display counts on top-left
        y = 30
        for name, cnt in counts.items():
            cv2.putText(annotated_frame, f"{name}: {cnt}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            y += 30

        cv2.imshow("2-item detector", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()