import json
import csv
import os

def convert_json_to_csv(json_file_path, csv_file_path):
    """
    Convert a JSON file to a CSV file with columns: index, Text, Label_A, Label_B
    
    Args:
        json_file_path (str): Path to the JSON file
        csv_file_path (str): Path to the output CSV file
    """
    # Create directory for the CSV file if it doesn't exist
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
    
    # Read the JSON file
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    
    # Write to CSV file
    with open(csv_file_path, 'w', encoding='utf-8', newline='') as csv_file:
        # Define the CSV writer with the specified columns
        csv_writer = csv.writer(csv_file)
        
        # Write the header
        csv_writer.writerow(['index', 'Text', 'Label_A', 'Label_B'])
        
        # Write each row from the JSON data
        for item in data:
            csv_writer.writerow([
                item.get('index', ''),
                item.get('Text', ''),
                item.get('Label_A', ''),
                item.get('Label_B', '')
            ])
    
    print(f"Conversion complete. CSV file saved at: {csv_file_path}")

# File paths
json_directory = 'dataset/scores.json'
csv_directory = 'dataset/scores.csv'

# Convert JSON to CSV
convert_json_to_csv(json_directory, csv_directory)
