#!/bin/bash

# Define variables
LOCAL_DIR="/Users/kapilwanaskar/Downloads/defactify_2024"
SCRIPT_NAME="text_baseline_4_full.py"
DATA_NAME="updated_test_data.csv"
ENV_NAME=".env"
REMOTE_DIR="~/defactify_2024"
TOTAL_CHUNKS=6  # 6 chunks total (2 per machine)
WORKERS_PER_CHUNK=4  # 4 workers per chunk
BATCH_SIZE=20
RATE_LIMIT=60

# Remote machine details
REMOTE_USER="016649880"
REMOTE1="10.31.96.168"
REMOTE2="10.31.96.185"

echo "=== Distributed Processing Setup ==="
echo "Total chunks: $TOTAL_CHUNKS (2 per machine)"
echo "Workers per chunk: $WORKERS_PER_CHUNK"
echo "Batch size: $BATCH_SIZE"
echo "Rate limit: $RATE_LIMIT requests/minute"

# Create directories on remote machines
echo -e "\n=== Setting up remote machines ==="
ssh $REMOTE_USER@$REMOTE1 "mkdir -p $REMOTE_DIR"
ssh $REMOTE_USER@$REMOTE2 "mkdir -p $REMOTE_DIR"

# Copy files to remote machines
echo -e "\n=== Copying files to remote machines ==="
scp $LOCAL_DIR/$SCRIPT_NAME $LOCAL_DIR/$DATA_NAME $LOCAL_DIR/$ENV_NAME $REMOTE_USER@$REMOTE1:$REMOTE_DIR/
scp $LOCAL_DIR/$SCRIPT_NAME $LOCAL_DIR/$DATA_NAME $LOCAL_DIR/$ENV_NAME $REMOTE_USER@$REMOTE2:$REMOTE_DIR/

# Install dependencies on remote machines
echo -e "\n=== Installing dependencies on remote machines ==="
ssh $REMOTE_USER@$REMOTE1 "cd $REMOTE_DIR && pip install pandas requests tqdm backoff python-dotenv"
ssh $REMOTE_USER@$REMOTE2 "cd $REMOTE_DIR && pip install pandas requests tqdm backoff python-dotenv"

# Start processing on remote machines using screen
echo -e "\n=== Starting processing on remote machines ==="
echo "Starting chunks 2-3 on $REMOTE1..."
ssh $REMOTE_USER@$REMOTE1 "cd $REMOTE_DIR && screen -dmS chunk2 python $SCRIPT_NAME --chunk 2 --total-chunks $TOTAL_CHUNKS --workers $WORKERS_PER_CHUNK --batch-size $BATCH_SIZE --rate-limit $RATE_LIMIT"
ssh $REMOTE_USER@$REMOTE1 "cd $REMOTE_DIR && screen -dmS chunk3 python $SCRIPT_NAME --chunk 3 --total-chunks $TOTAL_CHUNKS --workers $WORKERS_PER_CHUNK --batch-size $BATCH_SIZE --rate-limit $RATE_LIMIT"

echo "Starting chunks 4-5 on $REMOTE2..."
ssh $REMOTE_USER@$REMOTE2 "cd $REMOTE_DIR && screen -dmS chunk4 python $SCRIPT_NAME --chunk 4 --total-chunks $TOTAL_CHUNKS --workers $WORKERS_PER_CHUNK --batch-size $BATCH_SIZE --rate-limit $RATE_LIMIT"
ssh $REMOTE_USER@$REMOTE2 "cd $REMOTE_DIR && screen -dmS chunk5 python $SCRIPT_NAME --chunk 5 --total-chunks $TOTAL_CHUNKS --workers $WORKERS_PER_CHUNK --batch-size $BATCH_SIZE --rate-limit $RATE_LIMIT"

# Start processing on local machine in separate terminals
echo -e "\n=== Starting processing on local machine ==="
echo "Starting chunk 0 in this terminal..."
osascript -e "tell application \"Terminal\" to do script \"cd $LOCAL_DIR && python $SCRIPT_NAME --chunk 0 --total-chunks $TOTAL_CHUNKS --workers $WORKERS_PER_CHUNK --batch-size $BATCH_SIZE --rate-limit $RATE_LIMIT\""
echo "Starting chunk 1 in a new terminal..."
osascript -e "tell application \"Terminal\" to do script \"cd $LOCAL_DIR && python $SCRIPT_NAME --chunk 1 --total-chunks $TOTAL_CHUNKS --workers $WORKERS_PER_CHUNK --batch-size $BATCH_SIZE --rate-limit $RATE_LIMIT\""

echo -e "\n=== All processing jobs started ==="
echo "Local machine: Chunks 0-1 (check Terminal windows)"
echo "Remote machine 1 ($REMOTE1): Chunks 2-3 (use 'ssh $REMOTE_USER@$REMOTE1 screen -r chunk2' to check)"
echo "Remote machine 2 ($REMOTE2): Chunks 4-5 (use 'ssh $REMOTE_USER@$REMOTE2 screen -r chunk4' to check)"
echo -e "\nMonitoring commands:"
echo "  - Check remote screen sessions: ssh $REMOTE_USER@$REMOTE1 screen -ls"
echo "  - Attach to remote screen: ssh $REMOTE_USER@$REMOTE1 screen -r chunk2"
echo "  - Detach from screen: Ctrl+A, then D"
