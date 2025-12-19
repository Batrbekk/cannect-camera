"""
Camera Stream Handler
Manages connections to Raspberry Pi camera streams via RTSP
"""

import cv2
import threading
import queue
import time
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class CameraHandler:
    """Handles individual camera stream connection and frame buffering"""

    def __init__(self, camera_id: str, stream_url: str, fps: int = 15,
                 resolution: Tuple[int, int] = (1920, 1080), buffer_size: int = 30):
        self.camera_id = camera_id
        self.stream_url = stream_url
        self.fps = fps
        self.resolution = resolution
        self.buffer_size = buffer_size

        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.capture: Optional[cv2.VideoCapture] = None
        self.running = False
        self.thread: Optional[threading.Thread] = None

        self.reconnect_delay = 5  # seconds
        self.max_reconnect_attempts = 10

    def start(self):
        """Start camera stream capture in background thread"""
        if self.running:
            logger.warning(f"Camera {self.camera_id} already running")
            return

        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        logger.info(f"Started camera {self.camera_id}")

    def stop(self):
        """Stop camera stream capture"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        if self.capture:
            self.capture.release()
        logger.info(f"Stopped camera {self.camera_id}")

    def _connect(self) -> bool:
        """Establish connection to camera stream"""
        try:
            self.capture = cv2.VideoCapture(self.stream_url)

            # Set stream properties
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
            self.capture.set(cv2.CAP_PROP_FPS, self.fps)

            if not self.capture.isOpened():
                logger.error(f"Failed to open camera {self.camera_id} at {self.stream_url}")
                return False

            logger.info(f"Connected to camera {self.camera_id}")
            return True

        except Exception as e:
            logger.error(f"Error connecting to camera {self.camera_id}: {e}")
            return False

    def _capture_loop(self):
        """Main capture loop running in background thread"""
        reconnect_attempts = 0

        while self.running:
            # Connect or reconnect to camera
            if not self.capture or not self.capture.isOpened():
                if reconnect_attempts >= self.max_reconnect_attempts:
                    logger.error(f"Max reconnection attempts reached for {self.camera_id}")
                    break

                logger.info(f"Attempting to connect to camera {self.camera_id} (attempt {reconnect_attempts + 1})")

                if not self._connect():
                    reconnect_attempts += 1
                    time.sleep(self.reconnect_delay)
                    continue

                reconnect_attempts = 0

            # Read frame
            try:
                ret, frame = self.capture.read()

                if not ret:
                    logger.warning(f"Failed to read frame from camera {self.camera_id}")
                    if self.capture:
                        self.capture.release()
                    time.sleep(1)
                    continue

                # Add frame to queue (drop oldest if full)
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    # Drop oldest frame
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame)
                    except:
                        pass

                # Control frame rate
                time.sleep(1.0 / self.fps)

            except Exception as e:
                logger.error(f"Error reading frame from {self.camera_id}: {e}")
                if self.capture:
                    self.capture.release()
                time.sleep(1)

    def get_frame(self, timeout: float = 1.0) -> Optional[cv2.Mat]:
        """Get latest frame from camera buffer"""
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def is_alive(self) -> bool:
        """Check if camera stream is active"""
        return self.running and self.thread and self.thread.is_alive()


class MultiCameraHandler:
    """Manages multiple camera streams"""

    def __init__(self, camera_configs: dict):
        self.cameras = {}
        self._initialize_cameras(camera_configs)

    def _initialize_cameras(self, camera_configs: dict):
        """Initialize all cameras from config"""
        for vending_id, vending_config in camera_configs.items():
            for cam_key in ['camera_1', 'camera_2']:
                if cam_key in vending_config:
                    cam_config = vending_config[cam_key]
                    camera_id = cam_config['id']

                    handler = CameraHandler(
                        camera_id=camera_id,
                        stream_url=cam_config['stream_url'],
                        fps=cam_config.get('fps', 15),
                        resolution=tuple(cam_config.get('resolution', [1920, 1080]))
                    )

                    self.cameras[camera_id] = {
                        'handler': handler,
                        'vending_id': vending_id,
                        'position': cam_config.get('position', 'unknown'),
                        'station_id': vending_config.get('station_id'),
                        'location': vending_config.get('location')
                    }

        logger.info(f"Initialized {len(self.cameras)} cameras")

    def start_all(self):
        """Start all camera streams"""
        for camera_id, camera_data in self.cameras.items():
            camera_data['handler'].start()
        logger.info("All cameras started")

    def stop_all(self):
        """Stop all camera streams"""
        for camera_id, camera_data in self.cameras.items():
            camera_data['handler'].stop()
        logger.info("All cameras stopped")

    def get_frame(self, camera_id: str) -> Optional[cv2.Mat]:
        """Get latest frame from specific camera"""
        if camera_id in self.cameras:
            return self.cameras[camera_id]['handler'].get_frame()
        return None

    def get_all_frames(self) -> dict:
        """Get latest frames from all cameras"""
        frames = {}
        for camera_id, camera_data in self.cameras.items():
            frame = camera_data['handler'].get_frame(timeout=0.1)
            if frame is not None:
                frames[camera_id] = {
                    'frame': frame,
                    'vending_id': camera_data['vending_id'],
                    'station_id': camera_data['station_id'],
                    'position': camera_data['position'],
                    'location': camera_data['location']
                }
        return frames

    def get_camera_status(self) -> dict:
        """Get status of all cameras"""
        status = {}
        for camera_id, camera_data in self.cameras.items():
            status[camera_id] = {
                'alive': camera_data['handler'].is_alive(),
                'vending_id': camera_data['vending_id'],
                'station_id': camera_data['station_id']
            }
        return status
