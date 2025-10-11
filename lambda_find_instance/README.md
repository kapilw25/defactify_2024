# Lambda Labs A10 GPU Auto-Reservation

Automated monitoring system that polls Lambda Labs API for A10 GPU availability in us-east-1 (Virginia) and automatically launches instances with email notifications.

## Features

- ✅ Auto-launches A10 GPU when available
- ✅ Email alerts to kapilw25@gmail.com
- ✅ Respects official rate limits (15s polling, 12s launch cooldown)
- ✅ Mounts DiskUsEast1 filesystem automatically
- ✅ Logs all events to SQLite database
- ✅ Encrypted secrets with git-secret

## Prerequisites

- Python 3.7+
- Lambda Labs API key
- Gmail account (for email notifications)
- GPG key (for git-secret encryption)

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Email (Gmail App Password)

1. Go to Google Account → Security → 2-Step Verification
2. Scroll to "App passwords" → Generate new app password
3. Copy the 16-character password

### 3. Update Secrets

Edit `config/.env`:
```bash
# Email settings
SMTP_USERNAME=kapilw25@gmail.com
SMTP_PASSWORD=<your-gmail-app-password>

# Lambda settings (already configured)
SSH_KEY_NAME=<your-lambda-ssh-key-name>
```

### 4. Encrypt Secrets (Optional but Recommended)

```bash
# Install git-secret
brew install git-secret  # macOS
# or
sudo apt-get install git-secret  # Linux

# Run setup
./setup_git_secret.sh
```

### 5. Run Monitor

```bash
cd /Users/kapilwanaskar/Downloads/research_projects/lambda_find_instance
python src/m01_lambda_gpu_monitor.py
```

## How It Works

1. **Verify Filesystem**: Checks that `DiskUsEast1` exists in us-east-1
2. **Poll API**: Checks availability every 15 seconds
3. **Launch**: When available, waits 12s (rate limit), then launches
4. **Email**: Sends notification with instance IP and SSH command
5. **Log**: Records all events to `outputs/centralized.db`

## Output Example

```
============================================================
Lambda Labs A10 GPU Auto-Reservation Monitor
============================================================

[1/2] Verifying filesystem...
  ✓ Filesystem verified: DiskUsEast1 (fs123abc)

[2/2] Starting monitor (interval: 15s)
  Region: us-east-1
  Instance: gpu_1x_a10
  Filesystem: DiskUsEast1
  Started: 2025-10-09 17:30:00

------------------------------------------------------------

[1] 17:30:00 - Checking availability...
  ❌ No capacity in us-east-1

[2] 17:30:15 - Checking availability...
  🎯 A10 AVAILABLE! Launching instance...
  ⏳ Instance launching: 0920582c7ff041399e34823a0be62549
  📧 Email sent: ✅ Lambda A10 GPU Launched

============================================================
✅ SUCCESS! Instance Launched
  Instance ID: 0920582c7ff041399e34823a0be62549
  IP Address: 198.51.100.2
  SSH: ssh ubuntu@198.51.100.2
============================================================
```

## Email Notification

You'll receive an email with:

```
✅ A10 GPU Instance Launched Successfully!

Instance Details:
- ID: 0920582c7ff041399e34823a0be62549
- IP Address: 198.51.100.2
- Region: us-east-1
- Filesystem: DiskUsEast1

Access:
- SSH: ssh ubuntu@198.51.100.2
- Jupyter: https://jupyter-xxx.lambdaspaces.com/?token=abc123

Cost: $0.60/hour

Action Required: Remember to terminate the instance when done!
```

## Database Queries

View monitoring history:

```bash
sqlite3 outputs/centralized.db "SELECT timestamp, attempt_number, availability_status, action_taken FROM lambda_gpu_monitoring ORDER BY id DESC LIMIT 10"
```

## Configuration

All settings in `config/.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `LAMBDA_API_KEY` | - | Your Lambda Labs API key |
| `REGION_NAME` | us-east-1 | Target region |
| `INSTANCE_TYPE` | gpu_1x_a10 | GPU instance type |
| `FILESYSTEM_NAME` | DiskUsEast1 | Filesystem to mount |
| `SSH_KEY_NAME` | - | Your SSH key name |
| `CHECK_INTERVAL` | 15 | Polling interval (seconds) |
| `LAUNCH_COOLDOWN` | 12 | Wait before launch (seconds) |
| `EMAIL_RECIPIENT` | kapilw25@gmail.com | Email recipient |
| `SMTP_SERVER` | smtp.gmail.com | SMTP server |
| `SMTP_PORT` | 587 | SMTP port |

## Rate Limits (Official)

- **General API**: 1 request/second
- **Launch endpoint**: 1 request/12 seconds (5/minute)

Monitor uses:
- 15-second polling (safe margin)
- 12-second cooldown before launch (exact limit)

## Troubleshooting

### Error: Filesystem not found
```bash
# List your filesystems
curl -u <API_KEY>: https://cloud.lambda.ai/api/v1/file-systems
```

### Error: Invalid SSH key
```bash
# List your SSH keys
curl -u <API_KEY>: https://cloud.lambda.ai/api/v1/ssh-keys
```

### Email not sending
- Use Gmail App Password (not regular password)
- Enable 2-Step Verification in Google Account
- Check `SMTP_USERNAME` and `SMTP_PASSWORD` in `.env`

## Unit Tests

```bash
python -m pytest unit_test/test_lambda_monitor.py -v
```

## Security Notes

- ⚠️ **Never commit `config/.env`** (it's in .gitignore)
- ✅ Use `git-secret` to encrypt secrets
- ✅ Use Gmail App Password (not main password)
- ✅ API key is encrypted in `config/.env.secret`

## Cost

- **A10 GPU**: $0.60/hour
- **Filesystem**: Billed per GB stored
- **Remember to terminate** instances when done!

## Terminating Instances

```bash
# Via dashboard
https://cloud.lambda.ai/instances

# Via API
curl -X POST https://cloud.lambda.ai/api/v1/instance-operations/terminate \
  -u <API_KEY>: \
  -d '{"instance_ids": ["<instance-id>"]}'
```

## Project Structure

```
lambda_find_instance/
├── src/
│   └── m01_lambda_gpu_monitor.py  # Main monitor script
├── config/
│   ├── .env                        # Secrets (gitignored)
│   └── .env.secret                 # Encrypted secrets
├── outputs/
│   └── centralized.db              # SQLite database
├── unit_test/
│   └── test_lambda_monitor.py      # Unit tests
├── plans/
│   └── plan1.md                    # Implementation plan
├── setup_git_secret.sh             # Secret encryption setup
├── requirements.txt
├── .gitignore
└── README.md
```

## License

MIT

## Author

Kapil Wanaskar (kapilw25@gmail.com)
