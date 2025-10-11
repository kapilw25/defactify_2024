#!/usr/bin/env python3
"""
Unit tests for Lambda GPU Monitor
Run: python -m pytest unit_test/test_lambda_monitor.py
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

# Mock environment before importing module
os.environ['LAMBDA_API_KEY'] = 'test_key'
os.environ['REGION_NAME'] = 'us-east-1'
os.environ['INSTANCE_TYPE'] = 'gpu_1x_a10'
os.environ['FILESYSTEM_NAME'] = 'DiskUsEast1'
os.environ['SSH_KEY_NAME'] = 'test-key'

from m01_lambda_gpu_monitor import (
    verify_filesystem,
    check_availability,
    launch_instance
)


class TestLambdaMonitor(unittest.TestCase):

    @patch('m01_lambda_gpu_monitor.requests.get')
    def test_verify_filesystem_success(self, mock_get):
        """Test filesystem verification with valid response"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {
                    "name": "DiskUsEast1",
                    "region": {"name": "us-east-1"},
                    "id": "fs123"
                }
            ]
        }
        mock_get.return_value = mock_response

        fs_id = verify_filesystem()
        self.assertEqual(fs_id, "fs123")

    @patch('m01_lambda_gpu_monitor.requests.get')
    def test_verify_filesystem_not_found(self, mock_get):
        """Test filesystem verification when filesystem missing"""
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": []}
        mock_get.return_value = mock_response

        with self.assertRaises(Exception):
            verify_filesystem()

    @patch('m01_lambda_gpu_monitor.requests.get')
    def test_check_availability_true(self, mock_get):
        """Test availability check when GPU available"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": {
                "gpu_1x_a10": {
                    "regions_with_capacity_available": [
                        {"name": "us-east-1"}
                    ]
                }
            }
        }
        mock_get.return_value = mock_response

        self.assertTrue(check_availability())

    @patch('m01_lambda_gpu_monitor.requests.get')
    def test_check_availability_false(self, mock_get):
        """Test availability check when GPU unavailable"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": {
                "gpu_1x_a10": {
                    "regions_with_capacity_available": []
                }
            }
        }
        mock_get.return_value = mock_response

        self.assertFalse(check_availability())

    @patch('m01_lambda_gpu_monitor.requests.post')
    def test_launch_instance(self, mock_post):
        """Test instance launch"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": {
                "instance_ids": ["instance123"]
            }
        }
        mock_post.return_value = mock_response

        instance_id = launch_instance()
        self.assertEqual(instance_id, "instance123")


if __name__ == '__main__':
    unittest.main()
