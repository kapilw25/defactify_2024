#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified TextBaseLine script with improved debugging, timeout handling,
and processing of a small subset of data.
"""

import os
import requests
import json
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
import Levenshtein
import nltk
from dotenv import load_dotenv
import logging
import time
import signal
import sys

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

# Download NLTK resources
logger.info("Downloading NLTK resources")
nltk.download('punkt', quiet=True)

# Get API keys from environment variables
together_apikey = os.environ.get('TOGETHER_API_KEY')
firework_apikey = os.environ.get('FIREWORK_API_KEY')

# Check if API keys are available
if not together_apikey:
    logger.error("TOGETHER_API_KEY not found in environment variables")
    sys.exit(1)
if not firework_apikey:
    logger.warning("FIREWORK_API_KEY not found in environment variables. Some functionality may be limited.")

# Define models - using only one model for testing
together_ai_models = [
    "mistralai/Mistral-7B-Instruct-v0.1",  # Using just one model for testing
]

# Define input and output file paths
INPUT_FILE = "test_data.csv"  # Update this to your local input file path
OUTPUT_FILE = "llm_detection_results.csv"  # Local output file path

# Set a timeout for API calls (in seconds)
API_TIMEOUT = 30

# Define a timeout handler
def timeout_handler(signum, frame):
    raise TimeoutError("API call timed out")

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

    # Set timeout handler
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(API_TIMEOUT)
    
    try:
        start_time = time.time()
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        elapsed_time = time.time() - start_time
        logger.info(f"API call completed in {elapsed_time:.2f} seconds")
        
        # Cancel the alarm
        signal.alarm(0)
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            logger.error(f"TogetherAI Error: {response.status_code}, {response.text}")
            return None
    except TimeoutError:
        logger.error(f"TogetherAI API call timed out after {API_TIMEOUT} seconds")
        return None
    except Exception as e:
        logger.error(f"TogetherAI API call failed: {str(e)}")
        return None
    finally:
        # Ensure the alarm is canceled
        signal.alarm(0)

def fireworks(question, api_key=firework_apikey, model="accounts/yi-01-ai/models/yi-large"):
    logger.info(f"Making Fireworks API call with model: {model}")
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

    # Set timeout handler
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(API_TIMEOUT)
    
    try:
        start_time = time.time()
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        elapsed_time = time.time() - start_time
        logger.info(f"API call completed in {elapsed_time:.2f} seconds")
        
        # Cancel the alarm
        signal.alarm(0)
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            logger.error(f"Fireworks Error: {response.status_code}, {response.text}")
            return None
    except TimeoutError:
        logger.error(f"Fireworks API call timed out after {API_TIMEOUT} seconds")
        return None
    except Exception as e:
        logger.error(f"Fireworks API call failed: {str(e)}")
        return None
    finally:
        # Ensure the alarm is canceled
        signal.alarm(0)

def get_edit_distance(text1, text2):
    logger.debug("Calculating edit distance")
    tokens1 = word_tokenize(text1)
    tokens2 = word_tokenize(text2)
    joined1 = " ".join(tokens1)
    joined2 = " ".join(tokens2)
    distance = Levenshtein.distance(joined1, joined2)
    logger.debug(f"Edit distance: {distance}")
    return distance

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

    # Process using the Fireworks model
    if firework_apikey:
        logger.info("Processing with Fireworks model")
        new_text_fw = fireworks(sentence)
        if new_text_fw:
            edit_score_fw = get_edit_distance(sentence, new_text_fw)
            edit_distance_score.append(Score(edit_score_fw, new_text_fw, "Yi-Large"))
            logger.info(f"Model Yi-Large edit distance: {edit_score_fw}")

    return edit_distance_score

def main():
    logger.info("Starting text_baseline.py")
    
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

    # Process each text and split into sentences
    for text_idx, text in enumerate(texts):
        try:
            logger.info(f"Processing text {text_idx+1}: {text[:50]}...")
            sentences = sent_tokenize(text)
            logger.info(f"Split into {len(sentences)} sentences")
            
            # Limit to just 1 sentence for testing
            sentences = sentences[:1]
            logger.info(f"Limited to {len(sentences)} sentences for testing")
            
            for sent_idx, sentence in enumerate(sentences):
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
        except Exception as e:
            logger.error(f"Error tokenizing text {text_idx+1}: {str(e)}", exc_info=True)
            # If sentence tokenization fails, try processing the whole text as one sentence
            logger.info(f"Attempting to process text {text_idx+1} as a single sentence...")
            try:
                sentence = text
                results = detect_text(sentence)
                if results:
                    best_result = min(results, key=lambda x: x.edit_score)
                    output_data.append({
                        'sentence_number': f"{text_idx+1}.1",
                        'original_text': sentence,
                        'best_model': best_result.model,
                        'edit_distance': best_result.edit_score,
                        'regenerated_text': best_result.new_text
                    })
                    logger.info(f"Best model: {best_result.model}, Edit distance: {best_result.edit_score}")
            except Exception as e:
                logger.error(f"Error processing text {text_idx+1} as a single sentence: {str(e)}", exc_info=True)

    # Export to CSV
    if output_data:
        logger.info(f"Saving results to {OUTPUT_FILE}")
        results_df = pd.DataFrame(output_data)
        results_df.to_csv(OUTPUT_FILE, index=False)
        logger.info(f"Results saved to {OUTPUT_FILE}")
    else:
        logger.error("No results to save")

if __name__ == "__main__":
    main()
