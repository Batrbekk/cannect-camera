"""
Demographics Analyzer (Anonymized)
Estimates age and gender WITHOUT storing biometric data
Uses aggregated, anonymous statistics only
"""

import cv2
import numpy as np
from collections import defaultdict
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class DemographicsAnalyzer:
    """
    Analyzes demographics (age, gender) in an anonymized way

    IMPORTANT: No facial recognition, no biometric storage, no personal data
    Only aggregated statistics are kept
    """

    def __init__(self, enabled: bool = True, anonymize_data: bool = True):
        """
        Initialize demographics analyzer

        Args:
            enabled: Whether demographics analysis is enabled
            anonymize_data: Always True - ensures no biometric data is stored
        """
        self.enabled = enabled
        self.anonymize_data = anonymize_data  # Always True for privacy

        # Aggregated statistics only (no individual data)
        self.stats = {
            'age_groups': defaultdict(int),  # 0-18, 19-30, 31-45, 46-60, 60+
            'gender_distribution': defaultdict(int),  # M, F
            'total_analyzed': 0
        }

        # Note: In production, you would use a pre-trained model like:
        # - DeepFace
        # - Age-Gender-Estimator
        # - InsightFace
        # For this demo, we'll use placeholder logic
        self.age_gender_model = None

        logger.info(f"DemographicsAnalyzer initialized (anonymization: {anonymize_data})")

    def process_detections(self, detections: Dict, frame: np.ndarray) -> Dict:
        """
        Process person detections and estimate demographics

        Args:
            detections: Detection results with people bounding boxes
            frame: Input frame

        Returns:
            dict: Aggregated demographic statistics (NO individual data)
        """
        if not self.enabled:
            return {
                'age_distribution': {},
                'gender_distribution': {},
                'analyzed_count': 0
            }

        demographics_stats = {
            'age_distribution': defaultdict(int),
            'gender_distribution': defaultdict(int),
            'analyzed_count': 0
        }

        # Process each detected person
        for person in detections.get('people', []):
            bbox = person['bbox']

            # Extract face region (if visible)
            face_region = self._extract_face_region(frame, bbox)

            if face_region is not None:
                # Estimate age and gender
                age_group, gender = self._estimate_demographics(face_region)

                if age_group and gender:
                    # Update AGGREGATED statistics only
                    demographics_stats['age_distribution'][age_group] += 1
                    demographics_stats['gender_distribution'][gender] += 1
                    demographics_stats['analyzed_count'] += 1

                    # Update cumulative stats
                    self.stats['age_groups'][age_group] += 1
                    self.stats['gender_distribution'][gender] += 1
                    self.stats['total_analyzed'] += 1

        return demographics_stats

    def _extract_face_region(self, frame: np.ndarray, bbox: List[int]) -> Optional[np.ndarray]:
        """
        Extract face region from person bounding box

        Args:
            frame: Input frame
            bbox: Person bounding box [x1, y1, x2, y2]

        Returns:
            np.ndarray: Face region or None if not detected
        """
        try:
            x1, y1, x2, y2 = bbox

            # Estimate face region (upper portion of person bbox)
            person_height = y2 - y1
            face_y1 = y1
            face_y2 = int(y1 + person_height * 0.25)  # Top 25% of person

            face_x1 = max(0, x1)
            face_x2 = min(frame.shape[1], x2)
            face_y1 = max(0, face_y1)
            face_y2 = min(frame.shape[0], face_y2)

            if face_y2 <= face_y1 or face_x2 <= face_x1:
                return None

            face_region = frame[face_y1:face_y2, face_x1:face_x2]

            # Minimum face size check
            if face_region.shape[0] < 20 or face_region.shape[1] < 20:
                return None

            return face_region

        except Exception as e:
            logger.warning(f"Error extracting face region: {e}")
            return None

    def _estimate_demographics(self, face_region: np.ndarray) -> tuple:
        """
        Estimate age group and gender from face region

        DEMO IMPLEMENTATION: Returns random estimates
        In production, use a pre-trained model like DeepFace

        Args:
            face_region: Face image region

        Returns:
            tuple: (age_group, gender) or (None, None)
        """
        # TODO: In production, replace with actual model inference
        # Example using DeepFace:
        #
        # from deepface import DeepFace
        # result = DeepFace.analyze(face_region, actions=['age', 'gender'],
        #                          enforce_detection=False)
        # age = result['age']
        # gender = result['gender']

        # For demo: Use image properties as heuristic (not accurate!)
        try:
            # Simplified heuristic based on color distribution
            # THIS IS NOT ACCURATE - Use proper models in production
            hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
            brightness = np.mean(hsv[:, :, 2])

            # Placeholder age estimation
            if brightness > 150:
                age_group = '19-30'
            elif brightness > 120:
                age_group = '31-45'
            else:
                age_group = '46-60'

            # Placeholder gender estimation (random for demo)
            gender_value = np.mean(face_region)
            gender = 'M' if gender_value > 127 else 'F'

            return age_group, gender

        except Exception as e:
            logger.warning(f"Error estimating demographics: {e}")
            return None, None

    def get_statistics(self) -> Dict:
        """
        Get aggregated demographics statistics

        Returns:
            dict: Anonymized, aggregated demographics data
        """
        total = self.stats['total_analyzed']

        if total == 0:
            return {
                'total_analyzed': 0,
                'age_distribution': {},
                'gender_distribution': {},
                'age_percentages': {},
                'gender_percentages': {}
            }

        # Calculate percentages
        age_percentages = {
            age_group: (count / total) * 100
            for age_group, count in self.stats['age_groups'].items()
        }

        gender_percentages = {
            gender: (count / total) * 100
            for gender, count in self.stats['gender_distribution'].items()
        }

        return {
            'total_analyzed': total,
            'age_distribution': dict(self.stats['age_groups']),
            'gender_distribution': dict(self.stats['gender_distribution']),
            'age_percentages': age_percentages,
            'gender_percentages': gender_percentages
        }

    def reset(self):
        """Reset all statistics"""
        self.stats = {
            'age_groups': defaultdict(int),
            'gender_distribution': defaultdict(int),
            'total_analyzed': 0
        }
        logger.info("Demographics statistics reset")
