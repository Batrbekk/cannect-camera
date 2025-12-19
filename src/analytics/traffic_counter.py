"""
Traffic Counter
Detects and counts people and vehicles using YOLO
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class TrafficCounter:
    """Counts people and vehicles in video streams"""

    def __init__(self, model_path: str = 'yolov8n.pt', device: str = 'cpu',
                 confidence: float = 0.5, iou_threshold: float = 0.45):
        """
        Initialize traffic counter

        Args:
            model_path: Path to YOLO model
            device: 'cpu' or 'cuda'
            confidence: Detection confidence threshold
            iou_threshold: IOU threshold for NMS
        """
        self.model = YOLO(model_path)
        self.device = device
        self.confidence = confidence
        self.iou_threshold = iou_threshold

        # COCO class IDs for people and vehicles
        self.person_class = 0
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck

        # Tracking data
        self.tracks = defaultdict(lambda: {
            'positions': deque(maxlen=30),
            'class_id': None,
            'first_seen': None,
            'last_seen': None
        })

        self.counters = {
            'people': 0,
            'vehicles': 0,
            'total_people': set(),
            'total_vehicles': set()
        }

        logger.info(f"TrafficCounter initialized with {model_path} on {device}")

    def process_frame(self, frame: np.ndarray, camera_id: str) -> Dict:
        """
        Process single frame and detect people/vehicles

        Args:
            frame: Input frame (BGR)
            camera_id: Identifier for the camera

        Returns:
            dict: Detection results with counts and bounding boxes
        """
        # Run YOLO detection
        results = self.model.track(
            frame,
            conf=self.confidence,
            iou=self.iou_threshold,
            device=self.device,
            persist=True,
            classes=[self.person_class] + self.vehicle_classes
        )

        detections = {
            'people': [],
            'vehicles': [],
            'people_count': 0,
            'vehicle_count': 0,
            'camera_id': camera_id
        }

        if not results or len(results) == 0:
            return detections

        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        # Track IDs (if available)
        track_ids = None
        if hasattr(result.boxes, 'id') and result.boxes.id is not None:
            track_ids = result.boxes.id.cpu().numpy().astype(int)

        # Process each detection
        for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            detection_data = {
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': float(conf),
                'center': [float(center_x), float(center_y)]
            }

            # Classify as person or vehicle
            if class_id == self.person_class:
                detections['people'].append(detection_data)
                detections['people_count'] += 1

                if track_ids is not None:
                    track_id = track_ids[i]
                    self.counters['total_people'].add(f"{camera_id}_{track_id}")

            elif class_id in self.vehicle_classes:
                detections['vehicles'].append(detection_data)
                detections['vehicle_count'] += 1

                if track_ids is not None:
                    track_id = track_ids[i]
                    self.counters['total_vehicles'].add(f"{camera_id}_{track_id}")

        return detections

    def calculate_movement(self, track_id: int, current_pos: Tuple[float, float]) -> Optional[Dict]:
        """
        Calculate movement direction and speed for a tracked object

        Args:
            track_id: Unique track identifier
            current_pos: Current position (x, y)

        Returns:
            dict: Movement data (direction, speed) or None
        """
        track = self.tracks[track_id]
        track['positions'].append(current_pos)

        if len(track['positions']) < 5:
            return None

        # Calculate direction vector
        positions = list(track['positions'])
        start_pos = positions[0]
        end_pos = positions[-1]

        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]

        distance = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx) * 180 / np.pi

        # Classify direction
        if abs(angle) < 45:
            direction = 'right'
        elif abs(angle) > 135:
            direction = 'left'
        elif angle > 0:
            direction = 'down'
        else:
            direction = 'up'

        return {
            'direction': direction,
            'angle': float(angle),
            'distance': float(distance),
            'speed': float(distance / len(positions))  # pixels per frame
        }

    def get_statistics(self) -> Dict:
        """Get cumulative traffic statistics"""
        return {
            'total_unique_people': len(self.counters['total_people']),
            'total_unique_vehicles': len(self.counters['total_vehicles']),
            'current_people': self.counters['people'],
            'current_vehicles': self.counters['vehicles']
        }

    def reset_counters(self):
        """Reset all counters"""
        self.counters = {
            'people': 0,
            'vehicles': 0,
            'total_people': set(),
            'total_vehicles': set()
        }
        self.tracks.clear()
        logger.info("Traffic counters reset")
