#!/bin/bash

# Define variables
LOCAL_DIR="/Users/kapilwanaskar/Downloads/defactify_2024"
RESULTS_DIR="$LOCAL_DIR/combined_results"
REMOTE_USER="016649880"
REMOTE1="10.31.96.168"
REMOTE2="10.31.96.185"
REMOTE_DIR="~/defactify_2024"

# Create local directory for results
mkdir -p $RESULTS_DIR

echo "=== Collecting Results from All Machines ==="

# Copy results from remote machines
echo -e "\n=== Collecting results from remote machine 1 ($REMOTE1) ==="
scp $REMOTE_USER@$REMOTE1:$REMOTE_DIR/llm_detection_results_full_chunk*.csv $RESULTS_DIR/

echo -e "\n=== Collecting results from remote machine 2 ($REMOTE2) ==="
scp $REMOTE_USER@$REMOTE2:$REMOTE_DIR/llm_detection_results_full_chunk*.csv $RESULTS_DIR/

# Copy local results
echo -e "\n=== Collecting local results ==="
cp $LOCAL_DIR/llm_detection_results_full_chunk*.csv $RESULTS_DIR/

# Combine results
echo -e "\n=== Combining all results ==="
python3 - << EOF
import pandas as pd
import glob
import os

print("Looking for CSV files in '$RESULTS_DIR'...")
files = glob.glob('$RESULTS_DIR/llm_detection_results_full_chunk*.csv')
print(f"Found {len(files)} CSV files: {[os.path.basename(f) for f in files]}")

if files:
    print("Reading and combining CSV files...")
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            print(f"  - {os.path.basename(f)}: {len(df)} rows")
            dfs.append(df)
        except Exception as e:
            print(f"  - Error reading {os.path.basename(f)}: {str(e)}")
    
    if dfs:
        combined_df = pd.concat(dfs)
        output_path = '$LOCAL_DIR/llm_detection_results_full_combined.csv'
        combined_df.to_csv(output_path, index=False)
        print(f"\nSuccessfully combined {len(dfs)} files with {len(combined_df)} total rows")
        print(f"Combined results saved to: {output_path}")
    else:
        print("No valid data frames to combine")
else:
    print("No CSV files found to combine")
EOF

echo -e "\n=== Processing complete! ==="
echo "Check the combined results file: $LOCAL_DIR/llm_detection_results_full_combined.csv"
