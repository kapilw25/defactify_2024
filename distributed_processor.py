import sys
import os
import pandas as pd
import pickle
import time
import sqlite3
import concurrent.futures
import threading
import requests
from tqdm import tqdm
from collections import defaultdict
from functools import lru_cache
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# Import functions from baseline_benchmark_3.py
from baseline_benchmark_3 import detect_text, Score

# Get start and end indices from command line arguments
start_idx = int(sys.argv[1])
end_idx = int(sys.argv[2])
machine_id = sys.argv[3]

# Constants
INPUT_FILE = 'dataset/updated_test_data.csv'
OUTPUT_FILE = f'results/baseline_benchmark_33_results_{machine_id}.csv'
CHECKPOINT_FILE = f'checkpoint/checkpoint_{machine_id}.pkl'
BATCH_SIZE = 10  # Increased from 5 to 10
DB_FILE = f'results/benchmark_results_{machine_id}.db'
CSV_EXPORT_FREQUENCY = 10  # Reduced checkpoint frequency from 5 to 10 batches
MAX_WORKERS = 6  # MAX_WORKERS =6 if 8-core M1 machine, 
# MAX_WORKERS =12 # if 8 gb Nvidia GPU

# Setup connection pooling for API calls
session = requests.Session()
retries = Retry(total=5, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
session.mount('http://', HTTPAdapter(pool_connections=MAX_WORKERS, pool_maxsize=MAX_WORKERS*2, max_retries=retries))
session.mount('https://', HTTPAdapter(pool_connections=MAX_WORKERS, pool_maxsize=MAX_WORKERS*2, max_retries=retries))

# Thread-safe locks
db_lock = threading.Lock()
checkpoint_lock = threading.Lock()

def ensure_dir_exists(file_path):
    """
    Ensure directory exists for a given file path
    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

def save_checkpoint(processed_indices):
    """
    Save checkpoint data to restore processing if interrupted
    """
    with checkpoint_lock:
        checkpoint_data = {
            'processed_indices': processed_indices
        }
        
        ensure_dir_exists(CHECKPOINT_FILE)
        with open(CHECKPOINT_FILE, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        print(f"Checkpoint saved with {len(processed_indices)} processed texts")

def load_checkpoint():
    """
    Load checkpoint data to resume processing
    """
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'rb') as f:
            checkpoint_data = pickle.load(f)
        print(f"Checkpoint loaded with {len(checkpoint_data['processed_indices'])} processed texts")
        return checkpoint_data
    else:
        print("No checkpoint found, starting from beginning")
        return {'processed_indices': set()}

def setup_sqlite_db():
    """
    Set up SQLite database for storing results
    """
    # Create directory if it doesn't exist
    ensure_dir_exists(DB_FILE)
    
    # Create connection to database
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Create tables if they don't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text_index INTEGER,
        original_text TEXT,
        model TEXT,
        edit_score REAL,
        new_text TEXT,
        best_model INTEGER DEFAULT 0,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        machine_id TEXT
    )
    ''')
    
    # Create index on text_index for faster queries
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_text_index ON results (text_index)')
    
    conn.commit()
    conn.close()
    print(f"SQLite database set up at {DB_FILE}")

def save_to_sqlite(data, original_data):
    """
    Save results to SQLite database (thread-safe)
    """
    with db_lock:
        # Create connection to database
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Check if we're dealing with dataset results
        if isinstance(original_data, pd.DataFrame):
            # Dataset case - group by text index and model
            text_model_data = defaultdict(lambda: defaultdict(list))
            for item in data:
                text_model_data[item.text_index][item.model].append(item)
            
            # Batch insert data
            batch_data = []
            
            for text_idx, models_data in text_model_data.items():
                # Get the original text from the dataframe
                original_text = original_data[original_data['index'] == text_idx]['Text'].values[0]
                
                # Find the best model (lowest edit score)
                best_model = None
                lowest_edit_score = float('inf')
                
                for model_name, items in models_data.items():
                    if items:  # If we have results for this model
                        edit_score = items[0].edit_score
                        if edit_score < lowest_edit_score:
                            lowest_edit_score = edit_score
                            best_model = model_name
                
                # Prepare data for each model
                for model_name, items in models_data.items():
                    if items:  # If we have results for this model
                        is_best = 1 if model_name == best_model else 0
                        batch_data.append((
                            text_idx, original_text, model_name, items[0].edit_score, 
                            items[0].new_text, is_best, machine_id
                        ))
            
            # Execute batch insert
            if batch_data:
                cursor.executemany(
                    "INSERT INTO results (text_index, original_text, model, edit_score, new_text, best_model, machine_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    batch_data
                )
        
        # Commit changes and close connection
        conn.commit()
        conn.close()

def export_to_csv():
    """
    Export all results from SQLite database to CSV
    """
    with db_lock:
        # Create directory if it doesn't exist
        ensure_dir_exists(OUTPUT_FILE)
        
        # Connect to database
        conn = sqlite3.connect(DB_FILE)
        
        # Query to get all unique text indices
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT text_index FROM results ORDER BY text_index")
        text_indices = [row[0] for row in cursor.fetchall()]
        
        # Prepare data for CSV
        rows = []
        
        for text_idx in text_indices:
            # Get original text
            cursor.execute("SELECT original_text FROM results WHERE text_index = ? LIMIT 1", (text_idx,))
            original_text = cursor.fetchone()[0]
            
            # Get all models for this text
            cursor.execute("SELECT model, edit_score, new_text, best_model FROM results WHERE text_index = ?", (text_idx,))
            model_results = cursor.fetchall()
            
            # Create row with original text
            row = {'index': text_idx, 'Text': original_text}
            
            # Add best model
            best_model = None
            for model_name, edit_score, new_text, is_best in model_results:
                if is_best:
                    best_model = model_name
                    break
            
            row['best_LLM_model'] = best_model
            
            # Add each model's data
            for model_name, edit_score, new_text, _ in model_results:
                row[f'{model_name}_Edit_Score'] = edit_score
                row[f'{model_name}_New_Text'] = new_text
            
            rows.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        # Clean column names
        if not df.empty:
            # Move 'best_LLM_model' columns to 3rd position if it exists
            if 'best_LLM_model' in df.columns:
                cols = df.columns.tolist()
                cols.insert(2, cols.pop(cols.index('best_LLM_model')))
                df = df[cols]
            
            # Clean column names
            df.columns = [col.replace('/', ' ').replace('_', ' ').replace('-', ' ') for col in df.columns]
        
        # Save to CSV
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"All results exported to {OUTPUT_FILE}")
        
        # Close connection
        conn.close()

def process_text(text_row, processed_indices):
    """
    Process a single text (for parallel execution)
    """
    text_idx = text_row['index']
    text = text_row['Text']
    
    # Skip if already processed
    if text_idx in processed_indices:
        return None, text_idx
    
    print(f"\nProcessing text with index {text_idx}...")
    
    try:
        # Run the benchmark for this text
        data = detect_text(text, session=session)  # Pass the session for connection pooling
        
        # Add text_index to each item
        for item in data:
            item.text_index = text_idx
            
        return data, text_idx
    except Exception as e:
        print(f"Error processing text {text_idx}: {e}")
        return None, None

def process_batch_parallel(batch_df, processed_indices, shared_processed_indices):
    """
    Process a batch of texts in parallel
    """
    batch_results = []
    newly_processed = []
    
    # Create a copy of the dataframe for this thread
    batch_df_copy = batch_df.copy()
    
    # Filter out already processed texts
    unprocessed_rows = [row for _, row in batch_df_copy.iterrows() if row['index'] not in processed_indices]
    
    if not unprocessed_rows:
        return [], []
    
    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit tasks for each text in the batch
        future_to_idx = {
            executor.submit(process_text, row, processed_indices): row['index']
            for row in unprocessed_rows
        }
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                data, processed_idx = future.result()
                if data:
                    batch_results.extend(data)
                    newly_processed.append(processed_idx)
                    # Update shared processed indices
                    with checkpoint_lock:
                        shared_processed_indices.add(processed_idx)
            except Exception as e:
                print(f"Error processing text {idx}: {e}")
    
    return batch_results, newly_processed

def main_distributed():
    """
    Main function to run the distributed batch processing with parallel execution
    """
    print(f"Starting distributed processing on machine {machine_id}")
    print(f"Processing indices from {start_idx} to {end_idx}")
    
    # Set up SQLite database
    setup_sqlite_db()
    
    # Load checkpoint if exists
    checkpoint = load_checkpoint()
    processed_indices = checkpoint['processed_indices']
    
    # Load dataset from CSV
    print(f"Loading dataset from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    
    # Filter dataset to only include rows in the specified range
    df_subset = df[(df['index'] >= start_idx) & (df['index'] < end_idx)]
    total_rows = len(df_subset)
    print(f"Dataset subset loaded with {total_rows} texts to process")
    
    # Create progress bar
    with tqdm(total=total_rows, initial=len([i for i in processed_indices if start_idx <= i < end_idx])) as pbar:
        # Process dataset in batches
        batch_count = 0
        
        for i in range(0, total_rows, BATCH_SIZE):
            # Get batch
            batch_df = df_subset.iloc[i:i+BATCH_SIZE]
            
            # Skip batch if all texts already processed
            if all(row['index'] in processed_indices for _, row in batch_df.iterrows()):
                print(f"Skipping batch {i//BATCH_SIZE + 1} (already processed)")
                continue
            
            print(f"\nProcessing batch {i//BATCH_SIZE + 1} of {(total_rows-1)//BATCH_SIZE + 1}...")
            
            try:
                # Process batch in parallel
                batch_results, newly_processed = process_batch_parallel(batch_df, processed_indices, processed_indices)
                
                # Save batch results to SQLite
                if batch_results:
                    save_to_sqlite(batch_results, batch_df)
                    batch_count += 1
                
                # Export to CSV periodically
                if batch_count % CSV_EXPORT_FREQUENCY == 0:
                    print(f"Exporting results to CSV after {batch_count} batches...")
                    export_to_csv()
                
                # Save checkpoint
                if newly_processed:
                    save_checkpoint(processed_indices)
                
                # Update progress bar
                pbar.update(len(newly_processed))
                
            except KeyboardInterrupt:
                print("\nProcessing interrupted by user")
                save_checkpoint(processed_indices)
                print("Checkpoint saved. You can resume processing later.")
                return
            except Exception as e:
                print(f"\nError processing batch: {e}")
                save_checkpoint(processed_indices)
                print("Checkpoint saved due to error. You can resume processing later.")
                raise
    
    # Final export to CSV
    export_to_csv()
    
    print("\nDistributed processing completed for this machine!")
    print(f"Processed {len([i for i in processed_indices if start_idx <= i < end_idx])} texts out of {total_rows}")

if __name__ == "__main__":
    main_distributed()