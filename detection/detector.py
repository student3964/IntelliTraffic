"""
============================================================
IntelliTraffic – Vehicle Detection (YOLOv8)
============================================================
Wraps Ultralytics YOLOv8 for detecting vehicles and persons.
Uses pretrained COCO model (auto-downloads on first run).

Supported classes:
    0: person, 2: car, 3: motorcycle, 5: bus, 7: truck
"""

from ultralytics import YOLO
import numpy as np


class VehicleDetector:
    """
    YOLOv8-based vehicle and person detector.

    The detector filters YOLO results to only return
    relevant vehicle classes and person detections
    (needed for triple-riding analysis).
    """

    def __init__(self, config):
        """
        Initialize the YOLOv8 detector.

        Args:
            config: Config object with detection settings.
        """
        self.model_path = config.get("detection", "model_path", default="yolov8n.pt")
        self.confidence = config.get("detection", "confidence", default=0.4)
        self.iou_threshold = config.get("detection", "iou_threshold", default=0.5)
        self.device = config.get("detection", "device", default="cpu")
        self.vehicle_classes = config.get("detection", "vehicle_classes", default=[2, 3, 5, 7])
        self.person_class = config.get("detection", "person_class", default=0)
        self.class_names = config.get("detection", "class_names", default={
            0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"
        })

        # Combine all classes we want to detect
        self.target_classes = self.vehicle_classes + [self.person_class]

        # Load pretrained YOLOv8 model
        print(f"[Detector] Loading YOLOv8 model: {self.model_path}")
        self.model = YOLO(self.model_path)
        print(f"[Detector] Model loaded. Device: {self.device}")

    def detect(self, frame: np.ndarray) -> dict:
        """
        Run detection on a single frame.

        Args:
            frame: BGR image (numpy array from OpenCV).

        Returns:
            Dict with keys:
                "vehicles": list of vehicle detections
                "persons": list of person detections
            Each detection is a dict:
                {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float,
                    "class_id": int,
                    "class_name": str
                }
        """
        # Run YOLOv8 inference
        results = self.model(
            frame,
            conf=self.confidence,
            iou=self.iou_threshold,
            device=self.device,
            classes=self.target_classes,
            verbose=False  # Suppress per-frame logs
        )

        vehicles = []
        persons = []

        # Parse results (single image → results[0])
        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())

                    # Get class name from config mapping
                    # class_names keys might be strings from YAML
                    class_name = self.class_names.get(
                        class_id,
                        self.class_names.get(str(class_id), "unknown")
                    )

                    detection = {
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": confidence,
                        "class_id": class_id,
                        "class_name": class_name
                    }

                    # Sort into vehicles vs persons
                    if class_id == self.person_class:
                        persons.append(detection)
                    elif class_id in self.vehicle_classes:
                        vehicles.append(detection)

        return {
            "vehicles": vehicles,
            "persons": persons
        }

    def get_all_detections(self, frame: np.ndarray) -> list:
        """
        Get all detections (vehicles + persons) as a flat list.
        Useful for feeding directly into the tracker.

        Args:
            frame: BGR image.

        Returns:
            List of all detection dicts.
        """
        result = self.detect(frame)
        return result["vehicles"] + result["persons"]
