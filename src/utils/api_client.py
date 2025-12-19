"""
CANNECT.AI API Client
Sends analytics data to the CANNECT.AI backend
"""

import requests
import logging
from typing import Dict, Any, Optional
import time
from datetime import datetime

logger = logging.getLogger(__name__)


class CannectAPIClient:
    """Client for sending analytics data to CANNECT.AI API"""

    def __init__(self, base_url: str, auth_token: str, endpoints: dict):
        self.base_url = base_url.rstrip('/')
        self.auth_token = auth_token
        self.endpoints = endpoints

        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {auth_token}',
            'Content-Type': 'application/json'
        })

        self.retry_attempts = 3
        self.retry_delay = 2  # seconds

    def send_analytics(self, analytics_data: Dict[str, Any]) -> bool:
        """
        Send real-time analytics data to the API

        Args:
            analytics_data: Dictionary containing analytics metrics

        Returns:
            bool: True if successful, False otherwise
        """
        endpoint = self.endpoints['analytics']
        url = f"{self.base_url}{endpoint}"

        # Add timestamp if not present
        if 'timestamp' not in analytics_data:
            analytics_data['timestamp'] = datetime.utcnow().isoformat()

        for attempt in range(self.retry_attempts):
            try:
                response = self.session.post(url, json=analytics_data, timeout=10)

                if response.status_code in [200, 201]:
                    logger.debug(f"Analytics data sent successfully")
                    return True
                elif response.status_code == 401:
                    logger.error("Authentication failed - check API token")
                    return False
                elif response.status_code >= 500:
                    logger.warning(f"Server error {response.status_code}, retrying...")
                    time.sleep(self.retry_delay)
                    continue
                else:
                    logger.error(f"Failed to send analytics: {response.status_code} - {response.text}")
                    return False

            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout (attempt {attempt + 1}/{self.retry_attempts})")
                time.sleep(self.retry_delay)
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"Connection error (attempt {attempt + 1}/{self.retry_attempts}): {e}")
                time.sleep(self.retry_delay)
            except Exception as e:
                logger.error(f"Unexpected error sending analytics: {e}")
                return False

        logger.error(f"Failed to send analytics after {self.retry_attempts} attempts")
        return False

    def send_campaign_analytics(self, campaign_id: str, analytics_data: Dict[str, Any]) -> bool:
        """
        Send analytics data for a specific campaign

        Args:
            campaign_id: Campaign identifier
            analytics_data: Analytics metrics for the campaign

        Returns:
            bool: True if successful, False otherwise
        """
        endpoint = self.endpoints['campaign_analytics'].format(campaign_id=campaign_id)
        url = f"{self.base_url}{endpoint}"

        if 'timestamp' not in analytics_data:
            analytics_data['timestamp'] = datetime.utcnow().isoformat()

        try:
            response = self.session.post(url, json=analytics_data, timeout=10)

            if response.status_code in [200, 201]:
                logger.debug(f"Campaign analytics sent successfully for campaign {campaign_id}")
                return True
            else:
                logger.error(f"Failed to send campaign analytics: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Error sending campaign analytics: {e}")
            return False

    def batch_send_analytics(self, analytics_batch: list) -> Dict[str, int]:
        """
        Send multiple analytics records in batch

        Args:
            analytics_batch: List of analytics data dictionaries

        Returns:
            dict: Statistics with 'success' and 'failed' counts
        """
        stats = {'success': 0, 'failed': 0}

        for analytics_data in analytics_batch:
            if self.send_analytics(analytics_data):
                stats['success'] += 1
            else:
                stats['failed'] += 1

        logger.info(f"Batch send complete: {stats['success']} success, {stats['failed']} failed")
        return stats

    def health_check(self) -> bool:
        """Check if API is reachable"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
