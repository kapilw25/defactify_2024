import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import os
import sys
import importlib.util
from collections import Counter

# Import the db_utils module with numeric prefix
spec = importlib.util.spec_from_file_location("db_utils", "src/000_db_utils.py")
db_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(db_utils)
CentralizedDB = db_utils.CentralizedDB

def download_nltk_data():
    """Download required NLTK data"""
    import ssl

    # Set NLTK data path to outputs directory
    nltk_data_dir = os.path.join(os.getcwd(), 'outputs', 'nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    nltk.data.path.append(nltk_data_dir)

    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        print("✅ NLTK data already available")
    except LookupError:
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        print("📥 Downloading NLTK data...")
        nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
        nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)
        print("✅ NLTK data downloaded")

def preprocess_text(text):
    """Clean and preprocess text for word cloud generation"""
    if pd.isna(text) or text is None:
        return ""

    # Convert to string and lowercase
    text = str(text).lower()

    # Remove special characters, numbers, and extra whitespace
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word not in stop_words and len(word) > 2]

    return ' '.join(filtered_words)

def generate_wordcloud_for_column(df, column_name, output_dir):
    """Generate word cloud for a specific column"""
    print(f"🎨 Generating word cloud for: {column_name}")

    if column_name not in df.columns:
        print(f"❌ Column {column_name} not found")
        return None, None, 0

    # Combine all text from the column
    all_text = df[column_name].dropna().astype(str).apply(preprocess_text)
    combined_text = ' '.join(all_text)

    if not combined_text.strip():
        print(f"⚠️  No valid text found in {column_name}")
        return None, None, 0

    # Count words for statistics
    word_counts = Counter(combined_text.split())
    top_words = dict(word_counts.most_common(50))

    # Generate word cloud
    wordcloud = WordCloud(
        width=1200,
        height=800,
        background_color='white',
        max_words=200,
        relative_scaling=0.5,
        colormap='viridis'
    ).generate(combined_text)

    # Create figure
    plt.figure(figsize=(15, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud: {column_name}', fontsize=20, fontweight='bold', pad=20)

    # Save the word cloud
    safe_filename = column_name.replace('/', '_').replace(' ', '_')
    output_path = f"{output_dir}/wordcloud_{safe_filename}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✅ Word cloud saved: {output_path}")
    return output_path, top_words, len(word_counts)

def main():
    print("🚀 Starting Word Cloud Generation...")

    # Download required NLTK data
    download_nltk_data()

    # Initialize database
    db = CentralizedDB()

    # Output directory
    output_dir = "outputs/002_wordcloud_generator"
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Load dataset from centralized location
        print("📥 Loading dataset...")
        df = pd.read_csv("outputs/001_data_analysis/dataset.csv")
        print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

        # Define target columns
        target_columns = [
            'prompt', 'Human_story', 'gemma-2-9b', 'mistral-7B',
            'qwen-2-72B', 'llama-8B', 'accounts/yi-01-ai/models/yi-large', 'GPT_4-o'
        ]

        generated_files = []

        print(f"\n🎨 Generating word clouds for {len(target_columns)} columns...")

        # Generate word cloud for each column
        for column in target_columns:
            output_path, top_words, word_count = generate_wordcloud_for_column(df, column, output_dir)

            if output_path:
                # Save metadata to database
                db.save_wordcloud_metadata(column, output_path, top_words, word_count)
                generated_files.append(output_path)

        # Log successful execution
        db.log_script_execution(
            script_name="002_wordcloud_generator.py",
            status="SUCCESS",
            description=f"Generated {len(generated_files)} word clouds",
            output_files=generated_files
        )

        print(f"\n✅ Word cloud generation completed successfully!")
        print(f"📁 Outputs saved to: {output_dir}/")
        print(f"🗄️  Metadata saved to: outputs/centralized.db")
        print(f"📊 Generated {len(generated_files)} word clouds")

    except Exception as e:
        db.log_script_execution(
            script_name="002_wordcloud_generator.py",
            status="ERROR",
            description=f"Error: {str(e)}"
        )
        print(f"❌ Error: {e}")
        return False

    return True

if __name__ == "__main__":
    main()