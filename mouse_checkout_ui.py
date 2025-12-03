import cv2
from ultralytics import YOLO

# Price per mouse in rupees
PRICE_PER_MOUSE = 150

# CHANGE THIS if your model path is different
MODEL_PATH = "runs/detect/train/weights/best.pt"


def main():
    # Load your trained YOLO model (only "mouse" class)
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open webcam")
        return

    checkout_count = 0
    total_amount = 0

    print("Mouse Checkout Prototype (OpenCV)")
    print("Controls:")
    print("  - Press 'c' to CHECKOUT (freeze current count and compute bill)")
    print("  - Press 'q' to QUIT\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from webcam")
            break

        # Run YOLO on the current frame
        results = model(frame, verbose=False)
        boxes = results[0].boxes

        # Count how many mice are detected in THIS frame
        if boxes is not None:
            current_count = len(boxes)
        else:
            current_count = 0

        # Draw bounding boxes, etc.
        annotated_frame = results[0].plot()

        # Overlay text: current count, last checkout & total
        h, w, _ = annotated_frame.shape
        cv2.putText(
            annotated_frame,
            f"Current mice: {current_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            annotated_frame,
            f"Checkout: {checkout_count} | Total: Rs {total_amount}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 0, 0),
            2,
        )

        cv2.putText(
            annotated_frame,
            "Press 'c' = checkout, 'q' = quit",
            (10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        cv2.imshow("Mouse Checkout (OpenCV)", annotated_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            # Quit
            break
        elif key == ord("c"):
            # Checkout: freeze current count and compute bill
            checkout_count = current_count
            total_amount = checkout_count * PRICE_PER_MOUSE

            print("\n=== CHECKOUT DONE ===")
            print(f"  Mice detected: {checkout_count}")
            print(f"  Price per mouse: Rs {PRICE_PER_MOUSE}")
            print(f"  TOTAL: Rs {total_amount}")
            print("=====================\n")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()