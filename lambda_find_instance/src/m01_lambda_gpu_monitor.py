#!/usr/bin/env python3
"""
Lambda Labs GPU Auto-Reservation Monitor (Region Agnostic)
Polls Lambda API across ALL regions, launches instance when available
Uses existing filesystem in target region if available
"""

import requests
import time
import os
import sys
from datetime import datetime
from dotenv import load_dotenv


class Tee:
    """Custom class to write output to both terminal and log file (like Unix tee command)"""
    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log_file = log_file

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()  # Ensure immediate write to file

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

# Load configuration
load_dotenv('config/.env')

# API Configuration
BASE_URL = "https://cloud.lambda.ai/api/v1"
HEADERS = {"Authorization": f"Bearer {os.getenv('LAMBDA_API_KEY')}"}

# GPU options
GPU_OPTIONS = {
    'A': {
        'name': 'Costlier A100',
        'types': ['gpu_1x_a100_sxm4', 'gpu_1x_a100'],
        'desc': 'A100 (40GB SXM4), A100 (40GB PCIe)'
    },
    'B': {
        'name': 'Cheaper A10/A6000',
        'types': ['gpu_1x_a10', 'gpu_1x_a6000'],
        'desc': 'A10 (24GB), A6000 (48GB)'
    },
    'C': {
        'name': 'Premium GH200',
        'types': ['gpu_1x_gh200'],
        'desc': 'GH200 (96GB)'
    },
    'D': {
        'name': 'All GPUs',
        'types': ['gpu_1x_a10', 'gpu_1x_a100_sxm4', 'gpu_1x_a100', 'gpu_1x_a6000', 'gpu_1x_gh200'],
        'desc': 'A10, A100, A6000, GH200 (all types)'
    }
}

INSTANCE_TYPES = []  # Will be set based on user choice
SSH_KEY = os.getenv('SSH_KEY_NAME')
CHECK_INTERVAL = int(os.getenv('CHECK_INTERVAL', 15))
LAUNCH_COOLDOWN = int(os.getenv('LAUNCH_COOLDOWN', 12))

# ntfy.sh notification (no API key needed!)
NTFY_TOPIC = os.getenv('NTFY_TOPIC', 'lambda-gpu-alert-kapil')


def send_notification(title, message, priority="default"):
    """Send push notification via ntfy.sh - no API key needed!"""
    try:
        # Remove emojis from title and strip whitespace
        title_clean = title.encode('ascii', 'ignore').decode('ascii').strip()

        requests.post(
            f"https://ntfy.sh/{NTFY_TOPIC}",
            data=message.encode('utf-8'),
            headers={
                "Title": title_clean,
                "Priority": priority,
                "Tags": "computer,zap"
            }
        )
        print(f"  Notification sent to ntfy.sh/{NTFY_TOPIC}")
    except Exception as e:
        print(f"  Notification failed: {e}")


def init_log_file():
    """Create timestamped log file and return file handle for Tee"""
    os.makedirs('logs', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'logs/lambda_monitor_{timestamp}.log'

    # Open log file in write mode (will be kept open for Tee)
    log_file = open(log_filename, 'w', buffering=1)  # Line buffered

    return log_file, log_filename


def get_existing_filesystem(region):
    """Find an existing filesystem in the specified region"""
    try:
        resp = requests.get(f"{BASE_URL}/file-systems", headers=HEADERS)
        resp.raise_for_status()

        filesystems = resp.json()["data"]

        # Find any filesystem in the target region
        for fs in filesystems:
            if fs["region"]["name"] == region:
                fs_name = fs["name"]
                print(f"  Found existing filesystem: {fs_name}")
                return fs_name

        # No filesystem found in this region
        print(f"  No filesystem found in {region}, launching without filesystem")
        return None

    except Exception as e:
        print(f"  Warning: Could not check filesystems: {e}")
        return None


def check_availability(verbose=False):
    """Check GPU availability across all regions"""
    resp = requests.get(f"{BASE_URL}/instance-types", headers=HEADERS)
    resp.raise_for_status()

    data = resp.json()["data"]
    results = {}

    for instance_type in INSTANCE_TYPES:
        if instance_type in data:
            all_regions = data[instance_type]["regions_with_capacity_available"]
            price = data[instance_type]["instance_type"]["price_cents_per_hour"] / 100

            # Verbose: show raw API response
            if verbose:
                print(f"\n   API Response for {instance_type}:")
                print(f"   Price: ${price}/hour")
                print(f"   Raw regions data: {all_regions}")

            results[instance_type] = {
                'available_regions': all_regions,
                'price': price
            }

    return results


def launch_instance(instance_type, region, filesystem_name=None):
    """Launch GPU instance in specified region with optional filesystem"""
    payload = {
        "region_name": region,
        "instance_type_name": instance_type,
        "ssh_key_names": [SSH_KEY],
        "name": f"AutoLaunched-{instance_type}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    }

    # Add filesystem only if one exists
    if filesystem_name:
        payload["file_system_names"] = [filesystem_name]

    resp = requests.post(
        f"{BASE_URL}/instance-operations/launch",
        headers=HEADERS,
        json=payload
    )
    resp.raise_for_status()
    return resp.json()["data"]["instance_ids"][0]


def get_instance_details(instance_id):
    """Fetch instance IP and Jupyter URL"""
    resp = requests.get(f"{BASE_URL}/instances/{instance_id}", headers=HEADERS)
    resp.raise_for_status()
    data = resp.json()["data"]
    return data.get("ip"), data.get("jupyter_url")


def main():
    """Main monitoring loop"""
    global INSTANCE_TYPES

    print("=" * 60)
    print("Lambda Labs GPU Auto-Reservation Monitor")
    print("=" * 60)

    # Ask user which GPUs to search for
    print("\nWhich GPU instances do you want to search for?")
    print("-" * 60)
    print("A) Costlier A100 variants")
    print("   - A100 (40GB SXM4) - $1.29/hr")
    print("   - A100 (40GB PCIe) - $1.29/hr")
    print()
    print("B) Cheaper A10/A6000")
    print("   - A10 (24GB) - $0.75/hr")
    print("   - A6000 (48GB) - $0.80/hr")
    print()
    print("C) Premium GH200")
    print("   - GH200 (96GB) - $1.49/hr")
    print()
    print("D) All GPUs (A10, A100, A6000, GH200)")
    print("-" * 60)

    while True:
        choice = input("\nEnter your choice (A/B/C/D): ").strip().upper()
        if choice in GPU_OPTIONS:
            INSTANCE_TYPES = GPU_OPTIONS[choice]['types']
            print(f"\n✓ Selected: {GPU_OPTIONS[choice]['name']}")
            print(f"  Searching for: {GPU_OPTIONS[choice]['desc']}")
            break
        else:
            print("Invalid choice. Please enter A, B, C, or D.")

    # Initialize log file and set up Tee to capture all output
    log_file_handle, log_filename = init_log_file()
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_file_handle)

    attempt = 0

    print(f"\nLog file created: {log_filename}")
    print("(All terminal output will be captured in the log file)")

    try:
        # Start monitoring
        print(f"\n[1/2] Configuration:")
        print(f"  Checking: {', '.join(INSTANCE_TYPES)}")
        print(f"  Target Regions: ANY region with availability")
        print(f"  SSH Key: {SSH_KEY}")
        print(f"  Note: Will use existing filesystem if available in target region")

        print(f"\n[2/2] Starting monitor (polling every {CHECK_INTERVAL}s)")
        print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n" + "-" * 60)

        while True:
            attempt += 1
            timestamp = datetime.now().strftime('%H:%M:%S')
            print(f"\n[{attempt}] {timestamp} - Checking availability...", end=" ")

            try:
                # Show verbose output every 10th attempt to prove it's real
                verbose = (attempt % 10 == 1)
                results = check_availability(verbose=verbose)

                # Find all available GPU+region combinations
                available_options = []
                for instance_type, info in results.items():
                    for region_info in info['available_regions']:
                        available_options.append({
                            'instance_type': instance_type,
                            'region': region_info['name'],
                            'price': info['price']
                        })

                if available_options:
                    # Sort by price and choose cheapest
                    available_options.sort(key=lambda x: x['price'])
                    best_option = available_options[0]

                    instance_type = best_option['instance_type']
                    target_region = best_option['region']
                    price = best_option['price']

                    print(f"🎯 {instance_type.upper()} AVAILABLE in {target_region} (${price:.2f}/hr - cheapest)!")

                    # Send notification BEFORE launching
                    send_notification(
                        "🎯 Lambda GPU Available!",
                        f"{instance_type.upper()} available in {target_region}\nPrice: ${price:.2f}/hour\nLaunching now...",
                        priority="high"
                    )

                    # Check for existing filesystem in target region
                    print(f"  Checking for filesystem in {target_region}...")
                    fs_name = get_existing_filesystem(target_region)

                    # Respect launch rate limit
                    print(f"  ⏳ Waiting {LAUNCH_COOLDOWN}s (rate limit)...")
                    time.sleep(LAUNCH_COOLDOWN)

                    # Launch instance
                    print(f"  🚀 Launching {instance_type} in {target_region}...")
                    instance_id = launch_instance(instance_type, target_region, fs_name)

                    # Wait for details
                    print(f"  ⏳ Fetching instance details...")
                    time.sleep(5)
                    ip, jupyter = get_instance_details(instance_id)

                    # Send success notification
                    fs_info = f"\nFilesystem: {fs_name}" if fs_name else "\nFilesystem: None"
                    send_notification(
                        "✅ Instance Launched!",
                        f"{instance_type.upper()} launched successfully in {target_region}!\n\nInstance ID: {instance_id}\nIP: {ip or 'booting...'}\nRegion: {target_region}{fs_info}\nCost: ${price:.2f}/hour\n\nSSH: ssh ubuntu@{ip or 'pending'}",
                        priority="urgent"
                    )

                    # Print success
                    print("\n" + "=" * 60)
                    print(f"✅ SUCCESS! {instance_type.upper()} Launched")
                    print(f"  Instance ID: {instance_id}")
                    print(f"  Region: {target_region}")
                    print(f"  IP Address: {ip}")
                    print(f"  SSH: ssh ubuntu@{ip}")
                    print(f"  Jupyter: {jupyter}")
                    print(f"  Filesystem: {fs_name or 'None'}")
                    print(f"  Cost: ${price:.2f}/hour")
                    print("=" * 60)

                    break

                else:
                    # No capacity anywhere
                    print(f"❌ No capacity in ANY region")

                    for itype, info in results.items():
                        if info['available_regions']:
                            region_names = [r['name'] for r in info['available_regions']]
                            print(f"   {itype}: available in {', '.join(region_names)}")
                        else:
                            print(f"   {itype}: nowhere")

            except requests.HTTPError as e:
                # Handle API errors
                error_data = e.response.json().get("error", {})
                error_code = error_data.get("code", "unknown")
                error_msg = error_data.get("message", str(e))
                suggestion = error_data.get("suggestion", "")

                # Fatal errors - stop monitoring
                if error_code in ['global/quota-exceeded', 'global/invalid-api-key',
                                   'instance-operations/launch/file-system-in-wrong-region']:
                    print(f"\n❌ Fatal error: {error_code}")
                    print(f"   {error_msg}")
                    if suggestion:
                        print(f"   Suggestion: {suggestion}")
                    raise

                print(f"⚠️  {error_code}: {error_msg}")
                if suggestion:
                    print(f"   Suggestion: {suggestion}")

            except Exception as e:
                print(f"⚠️  Error: {e}")

            # Wait before next check
            time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        print("\n\n⏹  Stopped by user (Ctrl+C)")

    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")

    finally:
        # Print final summary (will be captured by Tee)
        print("\n" + "=" * 60)
        print(f"Session ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total attempts: {attempt}")
        print(f"Log file: {log_filename}")
        print("=" * 60)

        # Restore original stdout and close log file
        sys.stdout = original_stdout
        log_file_handle.close()


if __name__ == "__main__":
    main()
