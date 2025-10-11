#!/usr/bin/env python3
"""
Simple test: Check if we can connect to Lambda API and get A10 availability
"""

import requests
import os
import sys
from dotenv import load_dotenv

# Load API key
load_dotenv('config/.env')

BASE_URL = "https://cloud.lambda.ai/api/v1"
API_KEY = os.getenv('LAMBDA_API_KEY')
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

print("=" * 60)
print("Lambda Labs API Connection Test")
print("=" * 60)

# Test 1: Check API connection
print("\n[Test 1] Testing API connection...")
try:
    resp = requests.get(f"{BASE_URL}/instance-types", headers=HEADERS)
    resp.raise_for_status()
    print("✅ API connection successful")
except Exception as e:
    print(f"❌ API connection failed: {e}")
    sys.exit(1)

# Test 2: Check A10 availability
print("\n[Test 2] Checking A10 availability in us-east-1...")
try:
    data = resp.json()["data"]

    if "gpu_1x_a10" in data:
        instance_info = data["gpu_1x_a10"]
        print(f"✅ Found gpu_1x_a10")
        print(f"   Price: ${instance_info['instance_type']['price_cents_per_hour']/100:.2f}/hour")

        available_regions = instance_info["regions_with_capacity_available"]

        if available_regions:
            print(f"\n   Available in {len(available_regions)} region(s):")
            for region in available_regions:
                marker = "🎯" if region["name"] == "us-east-1" else "  "
                print(f"   {marker} {region['name']}: {region['description']}")
        else:
            print("   ❌ No capacity available in any region")

        # Check specifically for us-east-1
        us_east_available = any(r["name"] == "us-east-1" for r in available_regions)
        if us_east_available:
            print("\n🎯 A10 IS AVAILABLE IN US-EAST-1!")
        else:
            print("\n❌ A10 not available in us-east-1")
    else:
        print("❌ gpu_1x_a10 not found in instance types")

except Exception as e:
    print(f"❌ Error checking availability: {e}")
    sys.exit(1)

# Test 3: Check filesystem
print("\n[Test 3] Checking filesystem DiskUsEast1...")
try:
    resp = requests.get(f"{BASE_URL}/file-systems", headers=HEADERS)
    resp.raise_for_status()

    filesystems = resp.json()["data"]
    found = False

    for fs in filesystems:
        if fs["name"] == "DiskUsEast1":
            found = True
            print(f"✅ Found DiskUsEast1")
            print(f"   Region: {fs['region']['name']} ({fs['region']['description']})")
            print(f"   ID: {fs['id']}")
            print(f"   In use: {fs['is_in_use']}")

            if fs['region']['name'] != 'us-east-1':
                print(f"   ⚠️  WARNING: Filesystem is in {fs['region']['name']}, not us-east-1!")
            break

    if not found:
        print("❌ DiskUsEast1 filesystem not found")
        print("\nAvailable filesystems:")
        for fs in filesystems:
            print(f"   - {fs['name']} (region: {fs['region']['name']})")

except Exception as e:
    print(f"❌ Error checking filesystem: {e}")

print("\n" + "=" * 60)
print("Test complete!")
print("=" * 60)
