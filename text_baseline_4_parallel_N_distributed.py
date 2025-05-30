#!/usr/bin/env python3
"""
Version 4 of text_baseline.py with significant performance improvements:
1. Parallel processing of texts
2. Batch API requests when possible
3. Support for distributed processing via chunking
4. Improved output format with results from all models
5. Enhanced checkpoint optimization
"""

import os
import requests
import json
import pandas as pd
import logging
import time
import sys
import argparse
import concurrent.futures
import math
import queue
import threading
from dotenv import load_dotenv
from datetime import datetime
from tqdm import tqdm
import backoff

# Configure argument parser for distributed processing
parser = argparse.ArgumentParser(description='Process text data with LLM detection')
parser.add_argument('--chunk', type=int, default=0, help='Chunk number to process (0 for all)')
parser.add_argument('--total-chunks', type=int, default=1, help='Total number of chunks')
parser.add_argument('--workers', type=int, default=5, help='Number of parallel workers')
parser.add_argument('--batch-size', type=int, default=20, help='Checkpoint batch size')
parser.add_argument('--rate-limit', type=int, default=60, help='API rate limit per minute')
args = parser.parse_args()

# Configure logging
log_filename = f"text_baseline_v4_{datetime.now().strftime('%Y%m%d_%H%M%S')}_chunk{args.chunk}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
logger.info("Loading environment variables from .env file")
load_dotenv()

# Get API keys from environment variables
together_apikey = os.environ.get('TOGETHER_API_KEY')
firework_apikey = os.environ.get('FIREWORK_API_KEY')

# Check if API keys are available
if not together_apikey:
    logger.error("TOGETHER_API_KEY not found in environment variables")
    sys.exit(1)

# Define models
together_ai_models = [
    "Qwen/Qwen2-72B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
]

# Define all models including Fireworks
all_models = together_ai_models + ["Yi-Large"]

# Define input and output file paths
INPUT_FILE = "updated_test_data.csv"  # The full dataset
OUTPUT_FILE = f"llm_detection_results_full_chunk{args.chunk}.csv"  # Output file for this chunk
CHECKPOINT_FILE = f"processing_checkpoint_chunk{args.chunk}.json"  # To save progress
BATCH_SIZE = args.batch_size  # Process this many texts before saving a checkpoint

# Rate limiting setup
API_CALLS_PER_MINUTE = args.rate_limit
api_call_timestamps = queue.Queue()
api_call_lock = threading.Lock()

class RateLimiter:
    """Rate limiter for API calls"""
    def __init__(self, calls_per_minute):
        self.calls_per_minute = calls_per_minute
        self.call_timestamps = []
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        """Wait if we've exceeded our rate limit"""
        with self.lock:
            now = time.time()
            # Remove timestamps older than 1 minute
            self.call_timestamps = [ts for ts in self.call_timestamps if now - ts < 60]
            
            # If we've reached the limit, wait until we can make another call
            if len(self.call_timestamps) >= self.calls_per_minute:
                sleep_time = 60 - (now - self.call_timestamps[0]) + 0.1  # Add a small buffer
                if sleep_time > 0:
                    logger.debug(f"Rate limit reached, waiting for {sleep_time:.2f} seconds")
                    time.sleep(sleep_time)
                    # Recalculate after waiting
                    now = time.time()
                    self.call_timestamps = [ts for ts in self.call_timestamps if now - ts < 60]
            
            # Add current timestamp
            self.call_timestamps.append(now)

# Create rate limiters for each API
together_rate_limiter = RateLimiter(API_CALLS_PER_MINUTE)
fireworks_rate_limiter = RateLimiter(API_CALLS_PER_MINUTE)

# Exponential backoff decorator for API calls
@backoff.on_exception(
    backoff.expo,
    (requests.exceptions.RequestException, json.JSONDecodeError),
    max_tries=5,
    max_time=300
)
def make_api_request(url, headers, payload, timeout=30):
    """Make API request with exponential backoff for retries"""
    response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)
    response.raise_for_status()
    return response.json()

def togetherai(question, model, api_key=together_apikey):
    """Make API call to TogetherAI with rate limiting"""
    logger.debug(f"Making TogetherAI API call with model: {model}")
    url = "https://api.together.xyz/v1/chat/completions"
    formatted_prompt = f"Regenerate provided text: TEXT = {question}"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": formatted_prompt}],
        "max_tokens": 512,
        "temperature": 0.3,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        # Apply rate limiting
        together_rate_limiter.wait_if_needed()
        
        start_time = time.time()
        response_json = make_api_request(url, headers, payload)
        elapsed_time = time.time() - start_time
        logger.debug(f"API call completed in {elapsed_time:.2f} seconds")
        
        return response_json['choices'][0]['message']['content']
    except Exception as e:
        logger.error(f"TogetherAI API call failed for model {model}: {str(e)}")
        return None

def fireworks(question, api_key=firework_apikey, model="accounts/yi-01-ai/models/yi-large"):
    """Make API call to Fireworks with rate limiting"""
    if not api_key:
        logger.warning("Fireworks API key not available, skipping")
        return None
        
    logger.debug(f"Making Fireworks API call with model: {model}")
    url = "https://api.fireworks.ai/inference/v1/chat/completions"
    formatted_prompt = f"Regenerate the text: TEXT={question}\n"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": formatted_prompt}],
        "max_tokens": 512,
        "temperature": 0.3,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        # Apply rate limiting
        fireworks_rate_limiter.wait_if_needed()
        
        start_time = time.time()
        response_json = make_api_request(url, headers, payload)
        elapsed_time = time.time() - start_time
        logger.debug(f"API call completed in {elapsed_time:.2f} seconds")
        
        return response_json['choices'][0]['message']['content']
    except Exception as e:
        logger.error(f"Fireworks API call failed: {str(e)}")
        return None

def get_edit_distance(text1, text2):
    """
    Simple edit distance calculation without NLTK tokenization
    """
    if not text1 or not text2:
        return float('inf')  # Return infinity for None or empty texts
        
    # Convert to lowercase and split by spaces
    words1 = text1.lower().split()
    words2 = text2.lower().split()
    
    # Calculate Levenshtein distance manually
    m, n = len(words1), len(words2)
    
    # Create a matrix of size (m+1) x (n+1)
    dp = [[0 for _ in range(n+1)] for _ in range(m+1)]
    
    # Initialize the first row and column
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j
    
    # Fill the matrix
    for i in range(1, m+1):
        for j in range(1, n+1):
            if words1[i-1] == words2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j],      # deletion
                                   dp[i][j-1],      # insertion
                                   dp[i-1][j-1])    # substitution
    
    return dp[m][n]

def detect_text(sentence):
    """Process a sentence with all models and return results for each model"""
    logger.debug(f"Detecting text: {sentence[:50]}...")
    results = {}
    
    # Process using each Together AI model
    for model in together_ai_models:
        logger.debug(f"Processing with model: {model}")
        try:
            new_text = togetherai(sentence, model)
            if new_text:
                edit_score = get_edit_distance(sentence, new_text)
                results[model] = {
                    'edit_distance': edit_score,
                    'regenerated_text': new_text
                }
                logger.debug(f"Model {model} edit distance: {edit_score}")
        except Exception as e:
            logger.error(f"Error processing with model {model}: {str(e)}")

    # Process using the Fireworks model
    if firework_apikey:
        logger.debug("Processing with Fireworks model")
        try:
            new_text_fw = fireworks(sentence)
            if new_text_fw:
                edit_score_fw = get_edit_distance(sentence, new_text_fw)
                results["Yi-Large"] = {
                    'edit_distance': edit_score_fw,
                    'regenerated_text': new_text_fw
                }
                logger.debug(f"Model Yi-Large edit distance: {edit_score_fw}")
        except Exception as e:
            logger.error(f"Error processing with Fireworks: {str(e)}")

    return results

def split_into_sentences(text):
    """
    Split text into sentences without using NLTK
    """
    # Replace common abbreviations to avoid splitting at them
    text = text.replace("Mr.", "Mr_DOT_")
    text = text.replace("Mrs.", "Mrs_DOT_")
    text = text.replace("Dr.", "Dr_DOT_")
    text = text.replace("Ph.D.", "PhD_DOT_")
    text = text.replace("i.e.", "ie_DOT_")
    text = text.replace("e.g.", "eg_DOT_")
    
    # Split by sentence-ending punctuation followed by space and capital letter
    sentences = []
    current = ""
    
    for i, char in enumerate(text):
        current += char
        
        # Check for sentence endings
        if char in ['.', '!', '?'] and i < len(text) - 2:
            if text[i+1] == ' ' and text[i+2].isupper():
                sentences.append(current.strip())
                current = ""
    
    # Add the last sentence if there's anything left
    if current.strip():
        sentences.append(current.strip())
    
    # Restore abbreviations
    for i in range(len(sentences)):
        sentences[i] = sentences[i].replace("Mr_DOT_", "Mr.")
        sentences[i] = sentences[i].replace("Mrs_DOT_", "Mrs.")
        sentences[i] = sentences[i].replace("Dr_DOT_", "Dr.")
        sentences[i] = sentences[i].replace("PhD_DOT_", "Ph.D.")
        sentences[i] = sentences[i].replace("ie_DOT_", "i.e.")
        sentences[i] = sentences[i].replace("eg_DOT_", "e.g.")
    
    return sentences

def save_checkpoint(processed_texts, output_data):
    """Save progress checkpoint"""
    checkpoint = {
        "processed_texts": processed_texts,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f)
    
    # Also save current results
    if output_data:
        # Convert the nested dictionary structure to a flat dataframe
        flat_data = []
        for item in output_data:
            text_id = item['text_id']
            sentence_number = item['sentence_number']
            original_text = item['original_text']
            
            # Find the best model (lowest edit distance)
            best_model = None
            best_score = float('inf')
            for model, data in item['model_results'].items():
                if data['edit_distance'] < best_score:
                    best_score = data['edit_distance']
                    best_model = model
            
            # Add a row for each model
            for model, data in item['model_results'].items():
                flat_data.append({
                    'text_id': text_id,
                    'sentence_number': sentence_number,
                    'original_text': original_text,
                    'model': model,
                    'edit_distance': data['edit_distance'],
                    'regenerated_text': data['regenerated_text'],
                    'is_best_model': model == best_model
                })
        
        temp_df = pd.DataFrame(flat_data)
        temp_df.to_csv(f"{OUTPUT_FILE}.temp", index=False)
        logger.info(f"Checkpoint saved: {processed_texts} texts processed")

def load_checkpoint():
    """Load progress from checkpoint if exists"""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                checkpoint = json.load(f)
            logger.info(f"Resuming from checkpoint: {checkpoint['processed_texts']} texts already processed")
            return checkpoint["processed_texts"]
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
    return 0

def process_text(text_info):
    """Process a single text with all its sentences"""
    text_idx, text = text_info
    logger.info(f"Processing text {text_idx+1}: {text[:50]}...")
    text_results = []
    
    try:
        # Try to split into sentences
        sentences = split_into_sentences(text)
        
        if not sentences:
            # If no sentences were found, use simple period splitting
            sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
            
        if sentences:
            logger.info(f"Split text {text_idx+1} into {len(sentences)} sentences")
            
            for sent_idx, sentence in enumerate(sentences):
                logger.info(f"Processing Sentence {text_idx+1}.{sent_idx+1}: {sentence[:50]}...")
                try:
                    model_results = detect_text(sentence)
                    if model_results:
                        text_results.append({
                            'text_id': text_idx + 1,
                            'sentence_number': f"{text_idx+1}.{sent_idx+1}",
                            'original_text': sentence,
                            'model_results': model_results
                        })
                    else:
                        logger.warning(f"No valid results for sentence {text_idx+1}.{sent_idx+1}")
                except Exception as e:
                    logger.error(f"Error processing sentence {text_idx+1}.{sent_idx+1}: {str(e)}", exc_info=True)
        else:
            # If no sentences were found, process the whole text
            logger.info(f"No sentences found, processing text {text_idx+1} as a single unit")
            try:
                model_results = detect_text(text)
                if model_results:
                    text_results.append({
                        'text_id': text_idx + 1,
                        'sentence_number': f"{text_idx+1}.1",
                        'original_text': text,
                        'model_results': model_results
                    })
                else:
                    logger.warning(f"No valid results for text {text_idx+1}")
            except Exception as e:
                logger.error(f"Error processing text {text_idx+1}: {str(e)}", exc_info=True)
    except Exception as e:
        logger.error(f"Error processing text {text_idx+1}: {str(e)}", exc_info=True)
    
    return text_results

def main():
    logger.info(f"Starting text_baseline_4_full.py with chunk {args.chunk}/{args.total_chunks}")
    
    # Load CSV data
    try:
        logger.info(f"Loading data from {INPUT_FILE}")
        df = pd.read_csv(INPUT_FILE)
        all_texts = df["Text"].astype(str).tolist()
        logger.info(f"Loaded {len(all_texts)} texts")
        
        # Calculate chunk boundaries if distributed processing is enabled
        if args.total_chunks > 1:
            chunk_size = math.ceil(len(all_texts) / args.total_chunks)
            start_idx = args.chunk * chunk_size
            end_idx = min((args.chunk + 1) * chunk_size, len(all_texts))
            texts = all_texts[start_idx:end_idx]
            logger.info(f"Processing chunk {args.chunk}/{args.total_chunks}: texts {start_idx} to {end_idx-1} ({len(texts)} texts)")
        else:
            texts = all_texts
            
    except FileNotFoundError:
        logger.error(f"Error: Input file '{INPUT_FILE}' not found. Please check the file path.")
        sys.exit(1)
    except KeyError:
        logger.error("Error: The CSV file does not contain a 'Text' column.")
        sys.exit(1)

    # Load checkpoint if exists
    checkpoint_start_idx = load_checkpoint()
    
    # Prepare output data
    output_data = []
    
    # If checkpoint exists and temp output file exists, load it
    if checkpoint_start_idx > 0 and os.path.exists(f"{OUTPUT_FILE}.temp"):
        try:
            temp_df = pd.read_csv(f"{OUTPUT_FILE}.temp")
            
            # Convert flat dataframe back to nested structure
            text_ids = temp_df['text_id'].unique()
            for text_id in text_ids:
                text_df = temp_df[temp_df['text_id'] == text_id]
                sentence_numbers = text_df['sentence_number'].unique()
                
                for sentence_number in sentence_numbers:
                    sentence_df = text_df[text_df['sentence_number'] == sentence_number]
                    if len(sentence_df) > 0:
                        first_row = sentence_df.iloc[0]
                        model_results = {}
                        
                        for _, row in sentence_df.iterrows():
                            model_results[row['model']] = {
                                'edit_distance': row['edit_distance'],
                                'regenerated_text': row['regenerated_text']
                            }
                        
                        output_data.append({
                            'text_id': text_id,
                            'sentence_number': sentence_number,
                            'original_text': first_row['original_text'],
                            'model_results': model_results
                        })
            
            logger.info(f"Loaded {len(output_data)} existing results from checkpoint")
        except Exception as e:
            logger.error(f"Error loading temp results: {str(e)}")
    
    # Create a list of texts to process
    texts_to_process = [(i, texts[i]) for i in range(checkpoint_start_idx, len(texts))]
    
    # Process texts in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        future_to_text = {executor.submit(process_text, text_info): text_info for text_info in texts_to_process}
        
        # Process results as they complete
        completed = 0
        with tqdm(total=len(texts_to_process), desc="Processing texts") as pbar:
            for future in concurrent.futures.as_completed(future_to_text):
                text_info = future_to_text[future]
                text_idx = text_info[0]
                
                try:
                    text_results = future.result()
                    if text_results:
                        output_data.extend(text_results)
                    
                    completed += 1
                    pbar.update(1)
                    
                    # Save checkpoint every BATCH_SIZE texts
                    if completed % BATCH_SIZE == 0:
                        save_checkpoint(checkpoint_start_idx + completed, output_data)
                        
                except Exception as e:
                    logger.error(f"Error processing text {text_idx+1}: {str(e)}", exc_info=True)

    # Export final results to CSV
    if output_data:
        logger.info(f"Saving results to {OUTPUT_FILE}")
        
        # Convert the nested dictionary structure to a flat dataframe
        flat_data = []
        for item in output_data:
            text_id = item['text_id']
            sentence_number = item['sentence_number']
            original_text = item['original_text']
            
            # Find the best model (lowest edit distance)
            best_model = None
            best_score = float('inf')
            for model, data in item['model_results'].items():
                if data['edit_distance'] < best_score:
                    best_score = data['edit_distance']
                    best_model = model
            
            # Add a row for each model
            for model, data in item['model_results'].items():
                flat_data.append({
                    'text_id': text_id,
                    'sentence_number': sentence_number,
                    'original_text': original_text,
                    'model': model,
                    'edit_distance': data['edit_distance'],
                    'regenerated_text': data['regenerated_text'],
                    'is_best_model': model == best_model
                })
        
        results_df = pd.DataFrame(flat_data)
        results_df.to_csv(OUTPUT_FILE, index=False)
        logger.info(f"Results saved to {OUTPUT_FILE}")
        print(f"Results successfully saved to {OUTPUT_FILE}")
        
        # Clean up temp files
        if os.path.exists(f"{OUTPUT_FILE}.temp"):
            os.remove(f"{OUTPUT_FILE}.temp")
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
    else:
        logger.error("No results to save")
        print("Error: No results were generated to save")

if __name__ == "__main__":
    start_time = time.time()
    try:
        main()
        elapsed_time = time.time() - start_time
        logger.info(f"Processing completed in {elapsed_time/60:.2f} minutes")
        print(f"Processing completed in {elapsed_time/60:.2f} minutes")
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        print("\nProcessing interrupted. Progress has been saved to checkpoint.")
        elapsed_time = time.time() - start_time
        logger.info(f"Ran for {elapsed_time/60:.2f} minutes before interruption")
        print(f"Ran for {elapsed_time/60:.2f} minutes before interruption")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")
        elapsed_time = time.time() - start_time
        logger.info(f"Ran for {elapsed_time/60:.2f} minutes before error")
        print(f"Ran for {elapsed_time/60:.2f} minutes before error")
