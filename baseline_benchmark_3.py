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
from huggingface_hub import InferenceClient

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
    "mistralai/Mistral-7B-Instruct-v0.1",
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
]

# Define Gemma model
gemma_model = "google/gemma-3-27b-it"

class Score:
    def __init__(self, edit_score, new_text, model):
        self.edit_score = edit_score
        self.new_text = new_text
        self.model = model
        self.text_index = None  # Will be set when processing dataset

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
        "temperature": 0.05,
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
        "temperature": 0.05,
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

def gemma(question, api_key=hf_api_key, model="google/gemma-3-27b-it"):
    """
    Call Gemma model via Hugging Face Inference API to regenerate text
    """
    try:
        client = InferenceClient(
            provider="nebius",
            api_key=api_key,
        )
        
        formatted_prompt = f"Regenerate provided text: TEXT = {question}"
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": formatted_prompt
                }
            ],
            temperature=0.05,
            top_p=1,
            stream=False,
        )
        
        # Extract the output message
        output_message = response.choices[0].message.content
        return output_message
        
    except Exception as e:
        print(f"Error with Gemma model: {e}")
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
    
    # Process with Gemma model
    print("Processing with Gemma...")
    new_text = gemma(text)
    if new_text:
        edit_score = get_edit_distance(text, new_text)
        edit_distance_score.append(Score(edit_score, new_text, "Gemma-3-27b"))
        print(f"  Edit distance: {edit_score}")
    else:
        print("  Failed to get response from Gemma")
        
    return edit_distance_score

def save_to_csv(data, original_data, filename='output.csv'):
    """
    Save results to CSV file with model comparisons
    This function handles both single text and dataset cases
    """
    rows = []
    
    # Check if we're dealing with dataset results or single text
    if isinstance(original_data, pd.DataFrame):
        # Dataset case - group by text index and model
        text_model_data = defaultdict(lambda: defaultdict(list))
        for item in data:
            text_model_data[item.text_index][item.model].append(item)
            
        for text_idx, models_data in text_model_data.items():
            # Get the original text from the dataframe
            original_text = original_data[original_data['index'] == text_idx]['Text'].values[0]
            
            row = {'index': text_idx, 'Text': original_text}
            
            # Track the best model (lowest edit score) for this row
            best_model = None
            lowest_edit_score = float('inf')
            
            # Add each model's data in separate columns
            for model_name, items in models_data.items():
                if items:  # If we have results for this model
                    edit_score = items[0].edit_score
                    row[f'{model_name}_Edit_Score'] = edit_score
                    row[f'{model_name}_New_Text'] = items[0].new_text
                    
                    # Check if this model has the lowest edit score so far
                    if edit_score < lowest_edit_score:
                        lowest_edit_score = edit_score
                        best_model = model_name
                else:
                    # Fill with empty values if no results for this model
                    row[f'{model_name}_Edit_Score'] = None
                    row[f'{model_name}_New_Text'] = None
            
            # Add the best model column
            row['best_LLM_model'] = best_model
            rows.append(row)
    else:
        # Single text case - group by model
        original_text = original_data
        models_data = defaultdict(list)
        for item in data:
            models_data[item.model].append(item)

        max_entries = max([len(entries) for entries in models_data.values()])
        
        for i in range(max_entries):
            row = {'index': i, 'Text': original_text}
            
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
                    row[f'{model_name}_New_Text'] = None
            
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
    # Define input and output file paths
    INPUT_FILE = 'dataset/updated_test_data.csv'
    OUTPUT_FILE = 'results/baseline_benchmark_34_results.csv'
    
    # Load dataset from CSV
    print(f"Loading dataset from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    
    # Take only the first 5 rows for testing
    test_df = df.head(5)
    print(f"Processing {len(test_df)} texts from dataset")
    
    all_results = []
    
    # Process each text in the dataset
    for index, row in test_df.iterrows():
        text = row['Text']
        print(f"\nProcessing text {index+1} of {len(test_df)}...")
        
        # Run the benchmark for this text
        data = detect_text(text)
        
        # Add results to the collection
        for item in data:
            item.text_index = row['index']  # Store original index from dataset
        all_results.extend(data)
    
    # Save all results to CSV
    save_to_csv(all_results, test_df, OUTPUT_FILE)
    
    print("Benchmark completed!")

if __name__ == "__main__":
    main()