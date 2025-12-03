
AI-Powered Smart Billing System

This project implements a fully automated smart billing system that performs retail checkout using real-time object detection, object tracking, and dynamic billing. A camera captures live video of products, a YOLO-based model detects items instantly, and a billing engine automatically calculates the bill. The goal is to provide a fast, accurate, and cashier-less checkout experience for retail environments.

⸻

Features
	•	Real-time object detection using YOLOv8
	•	SORT-based object tracking to prevent double-counting
	•	Smart billing engine that maintains item quantities and totals
	•	Product database with SKU, price, and description
	•	Live dashboard/UI (Streamlit/Flask)
	•	Modular and scalable design

⸻

System Architecture

Camera → YOLO Detection → SORT Tracking → Billing Engine → UI Display

⸻

Project Structure

smart_billing/
├── data_prep/          (Dataset preparation scripts)
├── models/             (YOLO training code + weights)
├── tracking/           (SORT tracker implementation)
├── billing/            (Billing engine and cart logic)
├── db/                 (Product database and helpers)
├── ui/                 (Streamlit/Flask UI)
├── config/             (Configuration and logging)
└── tests/              (Unit tests)

⸻

Modules Overview

1. Data Preparation Module

Prepares YOLO dataset by resizing images, converting annotations, and splitting into train/val/test.
Technologies: opencv-python, numpy, pandas

2. Object Detection (YOLO Training)

Trains YOLOv8 on product images and exports best weights.
Technologies: ultralytics, torch

3. Real-time Inference Module

Runs YOLO on camera feed and displays detection results.
Technologies: ultralytics, opencv-python

4. SORT Tracking Module

Assigns unique IDs to objects and tracks them across frames to avoid double-counting.
Technologies: numpy, filterpy

5. Product Database Module

Stores and retrieves product details such as price and SKU.
Technologies: sqlite3, SQLAlchemy (optional)

6. Billing Engine Module

Updates quantities, calculates totals, and maintains the bill with 100% accuracy.
Technologies: decimal, pure Python

7. Backend / API Module (Optional)

Provides REST APIs for bill retrieval and system integration.
Technologies: Flask or FastAPI

8. User Interface (UI) Module

Displays live video feed, detected items, and the dynamic bill.
Technologies: Streamlit or Flask

9. Configuration & Logging

Handles paths, model configs, camera index, logging, and thresholds.
Technologies: logging, pyyaml

10. Testing & Evaluation

Unit tests for billing engine, tracking, DB, etc.
Technologies: pytest

⸻

Installation
	1.	Clone the repository
git clone https://github.com/yourusername/smart_billing.git
cd smart_billing
	2.	Create a virtual environment
python3 -m venv venv
source venv/bin/activate
	3.	Install dependencies
pip install -r requirements.txt

⸻

How to Run
	1.	Train YOLO Model
python models/train_yolo.py
	2.	Run Real-Time Detection + Billing
python inference.py
	3.	Start UI (Streamlit Example)
streamlit run ui/ui_streamlit.py

⸻

Output
	•	Live camera feed with bounding boxes
	•	Real-time product detection
	•	Automatic quantity counting
	•	Dynamic bill generation with totals

⸻

Future Improvements
	•	Multi-camera support
	•	Faster transformer-based detector models
	•	Mobile app version
	•	Cloud-based product catalog
	•	POS system integration

⸻

License

This project is released under the MIT License.

