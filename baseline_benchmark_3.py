#!/usr/bin/env python3
"""
Baseline Benchmark Script for Text Regeneration
This script evaluates different LLM models on their ability to regenerate text
by measuring the edit distance between original and regenerated text.
"""

import requests
import json
import os
import pandas as pd
import nltk
import Levenshtein
from dotenv import load_dotenv
from nltk.tokenize import word_tokenize
from collections import defaultdict

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment variables
together_apikey = os.environ.get('TOGETHER_API_KEY')
firework_apikey = os.environ.get('FIREWORK_API_KEY')
hf_api_key = os.environ.get('HF_API_KEY')

# Download NLTK punkt_tab data
nltk.download('punkt_tab')

# Define models to test
together_ai_models = [
    "Qwen/Qwen2-72B-Instruct",
    # "google/gemma-2-9b-it",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
]

class Score:
    def __init__(self, edit_score, new_text, model):
        self.edit_score = edit_score
        self.new_text = new_text
        self.model = model

def togetherai(question, model, api_key=together_apikey):
    """
    Call Together AI API to regenerate text using specified model
    """
    url = "https://api.together.xyz/v1/chat/completions"
    formatted_prompt = f"Regenerate provided text: TEXT = {question}"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": formatted_prompt}],
        "max_tokens": 1024,
        "top_p": 1,
        "top_k": 40,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "temperature": 0.5,
    }

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))

    if response.status_code == 200:
        # Extract the output message
        output_message = response.json()['choices'][0]['message']['content']
        return output_message
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

def fireworks(question, api_key=firework_apikey, model="accounts/yi-01-ai/models/yi-large"):
    """
    Call Fireworks AI API to regenerate text using specified model
    """
    url = "https://api.fireworks.ai/inference/v1/chat/completions"
    formatted_prompt = f"Regenerate the text: TEXT={question}\n"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": formatted_prompt}],
        "max_tokens": 1024,
        "top_p": 1,
        "top_k": 40,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "temperature": 0.5,
    }

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))

    if response.status_code == 200:
        # Extract the output message
        output_message = response.json()['choices'][0]['message']['content']
        return output_message
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

def get_edit_distance(text1, text2):
    """
    Calculate Levenshtein edit distance between two texts
    """
    tokens1 = word_tokenize(text1)
    tokens2 = word_tokenize(text2)
    joined1 = " ".join(tokens1)
    joined2 = " ".join(tokens2)
    distance = Levenshtein.distance(joined1, joined2)
    return distance

def detect_text(text):
    """
    Process text through multiple models and calculate edit distances
    """
    edit_distance_score = []
    
    # Process with Together AI models
    for model in together_ai_models:
        print(f"Processing with {model}...")
        new_text = togetherai(text, model)
        if new_text:
            edit_score = get_edit_distance(text, new_text)
            edit_distance_score.append(Score(edit_score, new_text, model))
            print(f"  Edit distance: {edit_score}")
        else:
            print(f"  Failed to get response from {model}")
    
    # Process with Fireworks AI
    print("Processing with Yi-Large...")
    new_text = fireworks(text)
    if new_text:
        edit_score = get_edit_distance(text, new_text)
        edit_distance_score.append(Score(edit_score, new_text, "Yi-Large"))
        print(f"  Edit distance: {edit_score}")
    else:
        print("  Failed to get response from Yi-Large")
        
    return edit_distance_score

def save_to_csv(data, original_text, filename='output.csv'):
    """
    Save results to CSV file with model comparisons
    """
    # Group data by model
    models_data = defaultdict(list)
    for item in data:
        models_data[item.model].append(item)

    # Create rows with model data in separate columns
    rows = []
    max_entries = max([len(entries) for entries in models_data.values()])
    
    for i in range(max_entries):
        row = {'Sr. No': i + 1, 'Original Text': original_text}
        
        # Track the best model (lowest edit score) for this row
        best_model = None
        lowest_edit_score = float('inf')
        
        # Add each model's data in separate columns
        for model_name, items in models_data.items():
            if i < len(items):
                edit_score = items[i].edit_score
                row[f'{model_name}_Edit_Score'] = items[i].edit_score
                row[f'{model_name}_New_Text'] = items[i].new_text
                
                # Check if this model has the lowest edit score so far
                if edit_score < lowest_edit_score:
                    lowest_edit_score = edit_score
                    best_model = model_name
            
            else:
                # Fill with empty values if this model has fewer entries
                row[f'{model_name}_Edit_Score'] = None
                row[f'{model_name}_Regenerated_Text'] = None
        
        # Add the best model column
        row['best_LLM_model'] = best_model
        rows.append(row)
    
    df = pd.DataFrame(rows)
    # Move 'best_LLM_model' columns to 3rd position
    cols = df.columns.tolist()
    cols.insert(2, cols.pop(cols.index('best_LLM_model')))
    df = df[cols]
    
    # Remove the file if it already exists
    if os.path.exists(filename):
        os.remove(filename)
        
    # Clean column names
    df.columns = [col.replace('/', ' ').replace('_', ' ').replace('-', ' ') for col in df.columns]
    
    # Save DataFrame to CSV
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

def main():
    """
    Main function to run the benchmark
    """
    # Sample text for testing
    text = """
    Global Economy Global Economy Supported by World Bank Sees Rosier Growth Outlook But rising trade barriers pose a long-term threat to global output as protectionist policies spread, the bank said. ByAlan Rappeport Reporting from Washington The World Bank on Tuesday raised its outlook for the world economy this year but warned that the rise of new trade barriers and protectionist policies posed a long-term threat to global growth. In its latest Global Economic Prospects report, the World Bank projected global growth to hold steady at 2.6 percent this year, an upgrade from itsJanuary forecastof 2.4 percent, and predicted that output would edge higher to 2.7 percent in 2025. The forecasts showed the global economy stabilizing after being rocked in recent years by the pandemic and the wars in Ukraine and the Middle East. "Four years after the upheavals caused by the pandemic, conflicts, inflation and monetary tightening, it appears that global economic growth is steadying," Indermit Gill, the World Bank's chief economist, said in a statement accompanying the report. However, sluggish growth continues to haunt the world's poorest economies, which are still grappling with inflation and the burdens of high debt. The bank noted that over the next three years, countries that account for more than 80 percent of the world's population would experience slower growth than in the decade before the pandemic. The slightly brighter forecast was led by the resilience of the U.S. economy, which continues to defy expectations despite higher interest rates. Overall, advanced economies are growing at an annual rate of 1.5 percent, with output remaining sluggish in Europe and Japan. By contrast, emerging market and developing economies are growing at a rate of 4 percent, led by China and Indonesia. Although growth is expected to be a bit stronger than previously forecast, the World Bank said prices were easing more slowly than it projected six months ago. It foresees global inflation moderating to 3.5 percent in 2024 and 2.9 percent next year. That gradual decline is likely to lead central banks to delay interest rate cuts, dimming prospects for growth in developing economies.
    """
    
    print("Starting text regeneration benchmark...")
    print(f"Processing text with {len(together_ai_models) + 1} models")
    
    # Run the benchmark
    data = detect_text(text)
    
    # Save results to CSV
    save_to_csv(data, text, 'baseline_benchmark_3_results.csv')

if __name__ == "__main__":
    main()
