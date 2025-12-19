"""
CANNECT.AI Computer Vision Analytics
Main processing script for 4-camera vending machine analytics
"""

import cv2
import yaml
import logging
import time
import os
import sys
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.camera_handler import MultiCameraHandler
from utils.api_client import CannectAPIClient
from analytics.traffic_counter import TrafficCounter
from analytics.attention_tracker import AttentionTracker
from analytics.demographics import DemographicsAnalyzer


def setup_logging(config: dict):
    """Configure logging"""
    log_config = config.get('logging', {})
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    log_file = log_config.get('file', 'logs/cv_analytics.log')

    # Create logs directory
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)


def load_config(config_path: str = 'config/config.yaml') -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class CVAnalyticsProcessor:
    """Main CV analytics processor"""

    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger

        # Initialize components
        self.logger.info("Initializing CV Analytics Processor...")

        # Camera handler
        self.camera_handler = MultiCameraHandler(config['cameras'])

        # Analytics modules
        model_config = config['model']
        self.traffic_counter = TrafficCounter(
            model_path=model_config['name'],
            device=model_config['device'],
            confidence=model_config['confidence'],
            iou_threshold=model_config['iou_threshold']
        )

        analytics_config = config['analytics']
        self.attention_tracker = AttentionTracker(
            dwell_time_threshold=analytics_config['attention_tracking']['dwell_time_threshold'],
            attention_zone_distance=analytics_config['attention_tracking']['attention_zone_distance']
        )

        self.demographics_analyzer = DemographicsAnalyzer(
            enabled=analytics_config['demographics']['enabled'],
            anonymize_data=analytics_config['demographics']['anonymize_data']
        )

        # API client
        api_config = config['api']
        auth_token = os.getenv('CANNECT_API_TOKEN', api_config.get('auth_token', ''))

        self.api_client = CannectAPIClient(
            base_url=api_config['base_url'],
            auth_token=auth_token,
            endpoints=api_config['endpoints']
        )

        # Processing config
        self.processing_config = config['processing']
        self.frame_skip = self.processing_config.get('frame_skip', 2)
        self.send_interval = self.processing_config.get('send_interval', 10)

        # State
        self.frame_count = 0
        self.last_send_time = time.time()
        self.running = False

        self.logger.info("CV Analytics Processor initialized successfully")

    def process_frame_batch(self, frames_data: dict) -> dict:
        """
        Process frames from all cameras

        Args:
            frames_data: Dictionary of frames from all cameras

        Returns:
            dict: Aggregated analytics results
        """
        analytics_results = {
            'timestamp': datetime.utcnow().isoformat(),
            'cameras': {},
            'aggregate': {
                'total_people': 0,
                'total_vehicles': 0,
                'total_engaged_viewers': 0,
                'demographics': {}
            }
        }

        current_time = time.time()

        # Process each camera
        for camera_id, cam_data in frames_data.items():
            frame = cam_data['frame']

            # Traffic counting
            detections = self.traffic_counter.process_frame(frame, camera_id)

            # Attention tracking
            attention_metrics = self.attention_tracker.process_detections(
                detections, current_time, frame.shape
            )

            # Demographics (anonymized)
            demographics = self.demographics_analyzer.process_detections(
                detections, frame
            )

            # Store results for this camera
            analytics_results['cameras'][camera_id] = {
                'station_id': cam_data['station_id'],
                'vending_id': cam_data['vending_id'],
                'location': cam_data['location'],
                'position': cam_data['position'],
                'traffic': {
                    'people_count': detections['people_count'],
                    'vehicle_count': detections['vehicle_count']
                },
                'attention': attention_metrics,
                'demographics': demographics
            }

            # Update aggregates
            analytics_results['aggregate']['total_people'] += detections['people_count']
            analytics_results['aggregate']['total_vehicles'] += detections['vehicle_count']
            analytics_results['aggregate']['total_engaged_viewers'] += attention_metrics['engaged_viewers']

        return analytics_results

    def send_analytics(self, analytics_data: dict):
        """Send analytics data to CANNECT.AI API"""
        try:
            success = self.api_client.send_analytics(analytics_data)
            if success:
                self.logger.info("Analytics data sent successfully")
            else:
                self.logger.warning("Failed to send analytics data")
        except Exception as e:
            self.logger.error(f"Error sending analytics: {e}")

    def run(self):
        """Main processing loop"""
        self.logger.info("Starting CV Analytics Processing...")

        # Start all cameras
        self.camera_handler.start_all()

        # Wait for cameras to initialize
        time.sleep(2)

        # Check camera status
        camera_status = self.camera_handler.get_camera_status()
        self.logger.info(f"Camera status: {camera_status}")

        self.running = True
        self.frame_count = 0

        try:
            while self.running:
                # Get frames from all cameras
                frames_data = self.camera_handler.get_all_frames()

                if not frames_data:
                    self.logger.warning("No frames received from cameras")
                    time.sleep(1)
                    continue

                # Skip frames for performance
                self.frame_count += 1
                if self.frame_count % self.frame_skip != 0:
                    continue

                # Process frames
                analytics_results = self.process_frame_batch(frames_data)

                # Send to API at intervals
                current_time = time.time()
                if current_time - self.last_send_time >= self.send_interval:
                    # Add cumulative statistics
                    analytics_results['cumulative'] = {
                        'traffic': self.traffic_counter.get_statistics(),
                        'attention': self.attention_tracker.get_statistics(),
                        'demographics': self.demographics_analyzer.get_statistics()
                    }

                    self.send_analytics(analytics_results)
                    self.last_send_time = current_time

                # Log progress
                if self.frame_count % 100 == 0:
                    self.logger.info(f"Processed {self.frame_count} frames from {len(frames_data)} cameras")

        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}", exc_info=True)
        finally:
            self.stop()

    def stop(self):
        """Stop processing and cleanup"""
        self.logger.info("Stopping CV Analytics Processor...")
        self.running = False
        self.camera_handler.stop_all()
        self.logger.info("CV Analytics Processor stopped")


def main():
    """Main entry point"""
    # Load environment variables
    load_dotenv()

    # Load configuration
    config_path = os.getenv('CONFIG_PATH', 'config/config.yaml')
    config = load_config(config_path)

    # Setup logging
    logger = setup_logging(config)
    logger.info("=" * 60)
    logger.info("CANNECT.AI CV Analytics Starting")
    logger.info("=" * 60)

    # Create and run processor
    processor = CVAnalyticsProcessor(config, logger)

    try:
        processor.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
