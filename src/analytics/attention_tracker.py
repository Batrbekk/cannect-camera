"""
Attention Tracker
Tracks viewer attention, dwell time, and engagement metrics
"""

import cv2
import numpy as np
from collections import defaultdict
import time
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class AttentionTracker:
    """Tracks viewer attention and engagement with displays"""

    def __init__(self, dwell_time_threshold: float = 2.0,
                 attention_zone_distance: float = 5.0):
        """
        Initialize attention tracker

        Args:
            dwell_time_threshold: Minimum time (seconds) to count as engaged viewer
            attention_zone_distance: Distance threshold for attention zone (meters/pixels)
        """
        self.dwell_time_threshold = dwell_time_threshold
        self.attention_zone_distance = attention_zone_distance

        # Track viewer engagement
        self.viewers = defaultdict(lambda: {
            'first_seen': None,
            'last_seen': None,
            'total_dwell_time': 0.0,
            'positions': [],
            'looking': False,
            'engaged': False
        })

        self.heatmap = None
        self.heatmap_size = (640, 480)  # Standard heatmap resolution

        logger.info("AttentionTracker initialized")

    def process_detections(self, detections: Dict, frame_time: float,
                           frame_shape: Tuple[int, int, int]) -> Dict:
        """
        Process detections and update attention metrics

        Args:
            detections: Detection results from traffic counter
            frame_time: Current frame timestamp
            frame_shape: Shape of input frame (H, W, C)

        Returns:
            dict: Attention metrics
        """
        if self.heatmap is None:
            self.heatmap = np.zeros(self.heatmap_size, dtype=np.float32)

        current_viewer_ids = set()
        attention_metrics = {
            'viewers_looking': 0,
            'engaged_viewers': 0,
            'avg_dwell_time': 0.0,
            'unique_viewers': len(self.viewers),
            'heatmap_updated': False
        }

        # Process each detected person
        for i, person in enumerate(detections.get('people', [])):
            viewer_id = f"{detections['camera_id']}_person_{i}"
            current_viewer_ids.add(viewer_id)

            viewer = self.viewers[viewer_id]

            # Initialize if first time seeing this viewer
            if viewer['first_seen'] is None:
                viewer['first_seen'] = frame_time

            viewer['last_seen'] = frame_time

            # Calculate dwell time
            dwell_time = frame_time - viewer['first_seen']
            viewer['total_dwell_time'] = dwell_time

            # Check if in attention zone (simplified - using center position)
            center = person['center']
            viewer['positions'].append(center)

            # Determine if looking at display (simplified heuristic)
            # In production, you'd use head pose estimation
            is_looking = self._estimate_attention(person, frame_shape)
            viewer['looking'] = is_looking

            # Check if engaged (looking + minimum dwell time)
            if is_looking and dwell_time >= self.dwell_time_threshold:
                viewer['engaged'] = True
                attention_metrics['engaged_viewers'] += 1

            if is_looking:
                attention_metrics['viewers_looking'] += 1

            # Update heatmap
            self._update_heatmap(center, frame_shape)

        # Clean up old viewers (not seen in last 30 seconds)
        self._cleanup_old_viewers(frame_time, current_viewer_ids)

        # Calculate average dwell time
        if len(self.viewers) > 0:
            total_dwell = sum(v['total_dwell_time'] for v in self.viewers.values())
            attention_metrics['avg_dwell_time'] = total_dwell / len(self.viewers)

        attention_metrics['unique_viewers'] = len(self.viewers)
        attention_metrics['heatmap_updated'] = True

        return attention_metrics

    def _estimate_attention(self, person_detection: Dict,
                           frame_shape: Tuple[int, int, int]) -> bool:
        """
        Estimate if person is paying attention to display

        This is a simplified heuristic. In production, use:
        - Head pose estimation (pitch, yaw, roll)
        - Gaze detection
        - Body orientation

        Args:
            person_detection: Person detection data
            frame_shape: Frame dimensions

        Returns:
            bool: True if estimated to be looking at display
        """
        # Simplified: Check if person is in central region of frame
        # This assumes camera is positioned to capture people looking at display
        center = person_detection['center']
        frame_height, frame_width = frame_shape[:2]

        # Define attention zone (center 60% of frame)
        zone_left = frame_width * 0.2
        zone_right = frame_width * 0.8
        zone_top = frame_height * 0.2
        zone_bottom = frame_height * 0.8

        in_zone = (zone_left <= center[0] <= zone_right and
                   zone_top <= center[1] <= zone_bottom)

        return in_zone

    def _update_heatmap(self, position: List[float], frame_shape: Tuple[int, int, int]):
        """
        Update attention heatmap with new position

        Args:
            position: (x, y) position in frame coordinates
            frame_shape: Original frame dimensions
        """
        frame_height, frame_width = frame_shape[:2]

        # Convert to heatmap coordinates
        heatmap_h, heatmap_w = self.heatmap_size
        hm_x = int(position[0] * heatmap_w / frame_width)
        hm_y = int(position[1] * heatmap_h / frame_height)

        # Ensure within bounds
        hm_x = max(0, min(hm_x, heatmap_w - 1))
        hm_y = max(0, min(hm_y, heatmap_h - 1))

        # Add Gaussian blob to heatmap
        sigma = 20
        x, y = np.meshgrid(np.arange(heatmap_w), np.arange(heatmap_h))
        gaussian = np.exp(-((x - hm_x)**2 + (y - hm_y)**2) / (2 * sigma**2))

        self.heatmap += gaussian * 0.1

        # Normalize to prevent overflow
        if self.heatmap.max() > 1000:
            self.heatmap = self.heatmap / self.heatmap.max() * 100

    def _cleanup_old_viewers(self, current_time: float, current_viewer_ids: set,
                            timeout: float = 30.0):
        """Remove viewers not seen recently"""
        to_remove = []

        for viewer_id, viewer in self.viewers.items():
            if viewer_id not in current_viewer_ids:
                if viewer['last_seen'] and (current_time - viewer['last_seen']) > timeout:
                    to_remove.append(viewer_id)

        for viewer_id in to_remove:
            del self.viewers[viewer_id]

    def get_heatmap(self, colormap: int = cv2.COLORMAP_JET) -> Optional[np.ndarray]:
        """
        Get attention heatmap visualization

        Args:
            colormap: OpenCV colormap to apply

        Returns:
            np.ndarray: Colored heatmap image or None
        """
        if self.heatmap is None:
            return None

        # Normalize heatmap
        if self.heatmap.max() > 0:
            normalized = (self.heatmap / self.heatmap.max() * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(self.heatmap, dtype=np.uint8)

        # Apply colormap
        colored_heatmap = cv2.applyColorMap(normalized, colormap)

        return colored_heatmap

    def get_statistics(self) -> Dict:
        """Get overall attention statistics"""
        engaged_count = sum(1 for v in self.viewers.values() if v['engaged'])
        looking_count = sum(1 for v in self.viewers.values() if v['looking'])

        total_dwell = sum(v['total_dwell_time'] for v in self.viewers.values())
        avg_dwell = total_dwell / len(self.viewers) if len(self.viewers) > 0 else 0.0

        return {
            'total_unique_viewers': len(self.viewers),
            'engaged_viewers': engaged_count,
            'viewers_currently_looking': looking_count,
            'average_dwell_time': avg_dwell,
            'viewing_rate': engaged_count / len(self.viewers) if len(self.viewers) > 0 else 0.0
        }

    def reset(self):
        """Reset all tracking data"""
        self.viewers.clear()
        self.heatmap = np.zeros(self.heatmap_size, dtype=np.float32)
        logger.info("Attention tracker reset")
