#!/usr/bin/env python3
"""
Simplified version of text_baseline.py that avoids using NLTK tokenization
"""

import os
import requests
import json
import pandas as pd
import logging
import time
import sys
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("text_baseline_debug.log"),
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

# Define models - using only one model for testing
together_ai_models = [
    "mistralai/Mistral-7B-Instruct-v0.1",  # Using just one model for testing
]

# Define input and output file paths
INPUT_FILE = "test_data.csv"  # Update this to your local input file path
OUTPUT_FILE = "llm_detection_results.csv"  # Local output file path

class Score:
    def __init__(self, edit_score, new_text, model):
        self.edit_score = edit_score
        self.new_text = new_text
        self.model = model

def togetherai(question, model, api_key=together_apikey):
    logger.info(f"Making TogetherAI API call with model: {model}")
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
        logger.info(f"API call completed in {elapsed_time:.2f} seconds")
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            logger.error(f"TogetherAI Error: {response.status_code}, {response.text}")
            return None
    except Exception as e:
        logger.error(f"TogetherAI API call failed: {str(e)}")
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
    logger.info(f"Detecting text: {sentence[:50]}...")
    edit_distance_score = []
    
    # Process using each Together AI model
    for model in together_ai_models:
        logger.info(f"Processing with model: {model}")
        new_text = togetherai(sentence, model)
        if new_text:
            edit_score = get_edit_distance(sentence, new_text)
            edit_distance_score.append(Score(edit_score, new_text, model))
            logger.info(f"Model {model} edit distance: {edit_score}")

    return edit_distance_score

def main():
    logger.info("Starting text_baseline_simple.py")
    
    # Load CSV data
    try:
        logger.info(f"Loading data from {INPUT_FILE}")
        df = pd.read_csv(INPUT_FILE)
        texts = df["Text"].astype(str).tolist()
        logger.info(f"Loaded {len(texts)} texts")
        
        # Limit to just 1 text for testing
        texts = texts[:1]
        logger.info(f"Limited to {len(texts)} texts for testing")
    except FileNotFoundError:
        logger.error(f"Error: Input file '{INPUT_FILE}' not found. Please check the file path.")
        sys.exit(1)
    except KeyError:
        logger.error("Error: The CSV file does not contain a 'Text' column.")
        sys.exit(1)

    # Prepare output data
    output_data = []

    # Process each text directly (no sentence tokenization)
    for text_idx, text in enumerate(texts):
        logger.info(f"Processing text {text_idx+1}: {text[:50]}...")
        try:
            # Simple sentence splitting by periods
            simple_sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
            logger.info(f"Split into {len(simple_sentences)} sentences")
            
            # Limit to just 1 sentence for testing
            if simple_sentences:
                simple_sentences = simple_sentences[:1]
                logger.info(f"Limited to {len(simple_sentences)} sentences for testing")
            
                for sent_idx, sentence in enumerate(simple_sentences):
                    logger.info(f"Processing Sentence {text_idx+1}.{sent_idx+1}: {sentence[:50]}...")
                    try:
                        results = detect_text(sentence)
                        if not results:
                            logger.warning("No valid results for this sentence.")
                            continue
                        best_result = min(results, key=lambda x: x.edit_score)
                        output_data.append({
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

    # Export to CSV
    if output_data:
        logger.info(f"Saving results to {OUTPUT_FILE}")
        results_df = pd.DataFrame(output_data)
        results_df.to_csv(OUTPUT_FILE, index=False)
        logger.info(f"Results saved to {OUTPUT_FILE}")
        print(f"Results successfully saved to {OUTPUT_FILE}")
    else:
        logger.error("No results to save")
        print("Error: No results were generated to save")

if __name__ == "__main__":
    main()
