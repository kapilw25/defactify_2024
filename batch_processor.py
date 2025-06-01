#!/usr/bin/env python3
"""
Batch Processor for Baseline Benchmark
This script processes the dataset in small batches, saves results in SQLite database,
and periodically exports to CSV for easy viewing. It also implements a checkpoint 
mechanism to resume processing if interrupted.
"""

import os
import pandas as pd
import pickle
import time
import sqlite3
from tqdm import tqdm
from collections import defaultdict

# Import functions from baseline_benchmark_3.py
from baseline_benchmark_3 import detect_text, Score

# Define models to test
together_ai_models = [
    "Qwen/Qwen2-72B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
]

# Define Gemma model
gemma_model = "google/gemma-3-27b-it"

# Constants
INPUT_FILE = 'dataset/updated_test_data.csv'
OUTPUT_FILE = 'results/baseline_benchmark_33_results.csv'
CHECKPOINT_FILE = 'checkpoint/checkpoint.pkl'
BATCH_SIZE = 3  # Number of texts to process in each batch
DB_FILE = 'results/benchmark_results.db'
CSV_EXPORT_FREQUENCY = 3  # Export to CSV after every 3 batches

def save_checkpoint(batch_results, processed_indices):
    """
    Save checkpoint data to restore processing if interrupted
    """
    checkpoint_data = {
        'batch_results': batch_results,
        'processed_indices': processed_indices
    }
    
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
        return {'batch_results': [], 'processed_indices': set()}

def setup_sqlite_db():
    """
    Set up SQLite database for storing results
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
    
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
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    conn.commit()
    conn.close()
    print(f"SQLite database set up at {DB_FILE}")

def save_to_sqlite(data, original_data):
    """
    Save results to SQLite database
    """
    # Create connection to database
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Check if we're dealing with dataset results
    if isinstance(original_data, pd.DataFrame):
        # Dataset case - group by text index and model
        text_model_data = defaultdict(lambda: defaultdict(list))
        for item in data:
            text_model_data[item.text_index][item.model].append(item)
            
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
            
            # Insert data for each model
            for model_name, items in models_data.items():
                if items:  # If we have results for this model
                    is_best = 1 if model_name == best_model else 0
                    cursor.execute(
                        "INSERT INTO results (text_index, original_text, model, edit_score, new_text, best_model) VALUES (?, ?, ?, ?, ?, ?)",
                        (text_idx, original_text, model_name, items[0].edit_score, items[0].new_text, is_best)
                    )
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    print(f"Batch results saved to {DB_FILE}")

def export_to_csv():
    """
    Export all results from SQLite database to CSV
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
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

def process_batch(batch_df, processed_indices):
    """
    Process a batch of texts
    """
    batch_results = []
    
    # Process each text in the batch
    for _, row in batch_df.iterrows():
        text_idx = row['index']
        text = row['Text']
        
        # Skip if already processed
        if text_idx in processed_indices:
            print(f"Skipping already processed text {text_idx}")
            continue
        
        print(f"\nProcessing text with index {text_idx}...")
        
        # Run the benchmark for this text
        data = detect_text(text)
        
        # Add results to the collection
        for item in data:
            item.text_index = text_idx
        batch_results.extend(data)
        
        # Mark as processed
        processed_indices.add(text_idx)
    
    return batch_results

def main():
    """
    Main function to run the batch processing
    """
    # Set up SQLite database
    setup_sqlite_db()
    
    # Load checkpoint if exists
    checkpoint = load_checkpoint()
    processed_indices = checkpoint['processed_indices']
    
    # Load dataset from CSV
    print(f"Loading dataset from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    total_rows = len(df)
    print(f"Dataset loaded with {total_rows} texts")
    
    # Create progress bar
    with tqdm(total=total_rows, initial=len(processed_indices)) as pbar:
        # Process dataset in batches
        batch_count = 0
        
        for i in range(0, total_rows, BATCH_SIZE):
            # Get batch
            batch_df = df.iloc[i:i+BATCH_SIZE]
            
            # Skip batch if all texts already processed
            if all(row['index'] in processed_indices for _, row in batch_df.iterrows()):
                print(f"Skipping batch {i//BATCH_SIZE + 1} (already processed)")
                continue
            
            print(f"\nProcessing batch {i//BATCH_SIZE + 1} of {(total_rows-1)//BATCH_SIZE + 1}...")
            
            try:
                # Process batch
                batch_results = process_batch(batch_df, processed_indices)
                
                # Save batch results to SQLite
                if batch_results:
                    save_to_sqlite(batch_results, batch_df)
                    batch_count += 1
                
                # Export to CSV periodically
                if batch_count % CSV_EXPORT_FREQUENCY == 0:
                    print(f"Exporting results to CSV after {batch_count} batches...")
                    export_to_csv()
                
                # Save checkpoint
                save_checkpoint([], processed_indices)  # We don't need to store results in checkpoint
                
                # Update progress bar
                pbar.update(len(batch_results) // len(together_ai_models + ['Yi-Large', 'Gemma-3-27b']))
                
                # Small delay to allow viewing results
                time.sleep(0.5)
                
            except KeyboardInterrupt:
                print("\nProcessing interrupted by user")
                save_checkpoint([], processed_indices)
                print("Checkpoint saved. You can resume processing later.")
                return
            except Exception as e:
                print(f"\nError processing batch: {e}")
                save_checkpoint([], processed_indices)
                print("Checkpoint saved due to error. You can resume processing later.")
                raise
    
    # Final export to CSV
    export_to_csv()
    
    print("\nBatch processing completed!")
    print(f"Processed {len(processed_indices)} texts out of {total_rows}")
    
    # Clean up checkpoint file if processing completed successfully
    if len(processed_indices) == total_rows and os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print("Checkpoint file removed as processing completed successfully")

if __name__ == "__main__":
    main()
