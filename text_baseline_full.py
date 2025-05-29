#!/usr/bin/env python3
"""
Production version of text_baseline.py that processes the full dataset
with improved error handling, progress tracking, and batch processing
"""

import os
import requests
import json
import pandas as pd
import logging
import time
import sys
from dotenv import load_dotenv
from datetime import datetime

# Configure logging
log_filename = f"text_baseline_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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

# Define input and output file paths
INPUT_FILE = "updated_test_data.csv"  # The full dataset
OUTPUT_FILE = "llm_detection_results_full.csv"  # Output file for full dataset
CHECKPOINT_FILE = "processing_checkpoint.json"  # To save progress
BATCH_SIZE = 10  # Process this many texts before saving a checkpoint

class Score:
    def __init__(self, edit_score, new_text, model):
        self.edit_score = edit_score
        self.new_text = new_text
        self.model = model

def togetherai(question, model, api_key=together_apikey):
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
        start_time = time.time()
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
        elapsed_time = time.time() - start_time
        logger.debug(f"API call completed in {elapsed_time:.2f} seconds")
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            logger.error(f"TogetherAI Error: {response.status_code}, {response.text}")
            return None
    except Exception as e:
        logger.error(f"TogetherAI API call failed: {str(e)}")
        return None

def fireworks(question, api_key=firework_apikey, model="accounts/yi-01-ai/models/yi-large"):
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
        start_time = time.time()
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
        elapsed_time = time.time() - start_time
        logger.debug(f"API call completed in {elapsed_time:.2f} seconds")
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            logger.error(f"Fireworks Error: {response.status_code}, {response.text}")
            return None
    except Exception as e:
        logger.error(f"Fireworks API call failed: {str(e)}")
        return None

def get_edit_distance(text1, text2):
    """
    Simple edit distance calculation without NLTK tokenization
    """
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
    logger.debug(f"Detecting text: {sentence[:50]}...")
    edit_distance_score = []
    
    # Process using each Together AI model
    for model in together_ai_models:
        logger.debug(f"Processing with model: {model}")
        new_text = togetherai(sentence, model)
        if new_text:
            edit_score = get_edit_distance(sentence, new_text)
            edit_distance_score.append(Score(edit_score, new_text, model))
            logger.debug(f"Model {model} edit distance: {edit_score}")

    # Process using the Fireworks model
    if firework_apikey:
        logger.debug("Processing with Fireworks model")
        new_text_fw = fireworks(sentence)
        if new_text_fw:
            edit_score_fw = get_edit_distance(sentence, new_text_fw)
            edit_distance_score.append(Score(edit_score_fw, new_text_fw, "Yi-Large"))
            logger.debug(f"Model Yi-Large edit distance: {edit_score_fw}")

    return edit_distance_score

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
        temp_df = pd.DataFrame(output_data)
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

def main():
    logger.info("Starting text_baseline_full.py")
    
    # Load CSV data
    try:
        logger.info(f"Loading data from {INPUT_FILE}")
        df = pd.read_csv(INPUT_FILE)
        texts = df["Text"].astype(str).tolist()
        logger.info(f"Loaded {len(texts)} texts")
    except FileNotFoundError:
        logger.error(f"Error: Input file '{INPUT_FILE}' not found. Please check the file path.")
        sys.exit(1)
    except KeyError:
        logger.error("Error: The CSV file does not contain a 'Text' column.")
        sys.exit(1)

    # Load checkpoint if exists
    start_idx = load_checkpoint()
    
    # Prepare output data
    output_data = []
    
    # If checkpoint exists and temp output file exists, load it
    if start_idx > 0 and os.path.exists(f"{OUTPUT_FILE}.temp"):
        try:
            temp_df = pd.read_csv(f"{OUTPUT_FILE}.temp")
            output_data = temp_df.to_dict('records')
            logger.info(f"Loaded {len(output_data)} existing results from checkpoint")
        except Exception as e:
            logger.error(f"Error loading temp results: {str(e)}")
    
    # Process each text
    total_texts = len(texts)
    for text_idx in range(start_idx, total_texts):
        text = texts[text_idx]
        logger.info(f"Processing text {text_idx+1}/{total_texts} ({(text_idx+1)/total_texts*100:.1f}%): {text[:50]}...")
        
        try:
            # Try to split into sentences
            sentences = split_into_sentences(text)
            
            if not sentences:
                # If no sentences were found, use simple period splitting
                sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
                
            if sentences:
                logger.info(f"Split into {len(sentences)} sentences")
                
                for sent_idx, sentence in enumerate(sentences):
                    logger.info(f"Processing Sentence {text_idx+1}.{sent_idx+1}: {sentence[:50]}...")
                    try:
                        results = detect_text(sentence)
                        if not results:
                            logger.warning("No valid results for this sentence.")
                            continue
                        best_result = min(results, key=lambda x: x.edit_score)
                        output_data.append({
                            'text_id': text_idx + 1,
                            'sentence_number': f"{text_idx+1}.{sent_idx+1}",
                            'original_text': sentence,
                            'best_model': best_result.model,
                            'edit_distance': best_result.edit_score,
                            'regenerated_text': best_result.new_text
                        })
                        logger.info(f"Best model: {best_result.model}, Edit distance: {best_result.edit_score}")
                    except Exception as e:
                        logger.error(f"Error processing sentence {text_idx+1}.{sent_idx+1}: {str(e)}", exc_info=True)
            else:
                # If no sentences were found, process the whole text
                logger.info(f"No sentences found, processing text {text_idx+1} as a single unit")
                try:
                    results = detect_text(text)
                    if results:
                        best_result = min(results, key=lambda x: x.edit_score)
                        output_data.append({
                            'text_id': text_idx + 1,
                            'sentence_number': f"{text_idx+1}.1",
                            'original_text': text,
                            'best_model': best_result.model,
                            'edit_distance': best_result.edit_score,
                            'regenerated_text': best_result.new_text
                        })
                        logger.info(f"Best model: {best_result.model}, Edit distance: {best_result.edit_score}")
                except Exception as e:
                    logger.error(f"Error processing text {text_idx+1}: {str(e)}", exc_info=True)
        except Exception as e:
            logger.error(f"Error processing text {text_idx+1}: {str(e)}", exc_info=True)
        
        # Save checkpoint every BATCH_SIZE texts
        if (text_idx + 1) % BATCH_SIZE == 0:
            save_checkpoint(text_idx + 1, output_data)

    # Export final results to CSV
    if output_data:
        logger.info(f"Saving results to {OUTPUT_FILE}")
        results_df = pd.DataFrame(output_data)
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
