"""
Test Camera Connections
Simple script to verify RTSP camera streams are accessible
"""

import cv2
import yaml
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def load_config():
    """Load camera configuration"""
    config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def test_camera_connection(camera_id: str, stream_url: str) -> bool:
    """
    Test individual camera connection

    Args:
        camera_id: Camera identifier
        stream_url: RTSP stream URL

    Returns:
        bool: True if connection successful
    """
    print(f"\nTesting {camera_id}: {stream_url}")
    print("-" * 60)

    try:
        cap = cv2.VideoCapture(stream_url)

        if not cap.isOpened():
            print(f"❌ Failed to open stream")
            return False

        # Try to read a frame
        ret, frame = cap.read()

        if not ret or frame is None:
            print(f"❌ Failed to read frame")
            cap.release()
            return False

        print(f"✅ Connection successful")
        print(f"   Frame size: {frame.shape[1]}x{frame.shape[0]}")
        print(f"   Channels: {frame.shape[2]}")

        cap.release()
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Test all cameras"""
    print("=" * 60)
    print("CANNECT.AI Camera Connection Test")
    print("=" * 60)

    config = load_config()

    results = {}

    # Test each camera
    for vending_id, vending_config in config['cameras'].items():
        print(f"\n📹 Testing {vending_id.upper()}")
        print(f"   Location: {vending_config['location']}")
        print(f"   Station: {vending_config['station_id']}")

        for cam_key in ['camera_1', 'camera_2']:
            if cam_key in vending_config:
                cam_config = vending_config[cam_key]
                camera_id = cam_config['id']
                stream_url = cam_config['stream_url']

                success = test_camera_connection(camera_id, stream_url)
                results[camera_id] = success

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    success_count = sum(results.values())
    total_count = len(results)

    for camera_id, success in results.items():
        status = "✅ OK" if success else "❌ FAILED"
        print(f"{camera_id}: {status}")

    print(f"\nTotal: {success_count}/{total_count} cameras connected")

    if success_count == total_count:
        print("\n🎉 All cameras are working!")
        return 0
    else:
        print(f"\n⚠️  {total_count - success_count} camera(s) failed")
        print("\nTroubleshooting:")
        print("1. Check Raspberry Pi is powered on")
        print("2. Verify network connectivity (ping IP)")
        print("3. Ensure RTSP server is running on Raspberry Pi")
        print("4. Check camera stream URLs in config/config.yaml")
        return 1


if __name__ == '__main__':
    sys.exit(main())
