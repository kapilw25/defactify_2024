import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
import os
import sys
import importlib.util
import nltk
import ssl
sys.path.append('.')

# Import the db_utils module with numeric prefix
spec = importlib.util.spec_from_file_location("db_utils", "src/000_db_utils.py")
db_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(db_utils)
CentralizedDB = db_utils.CentralizedDB

def download_nltk_data(output_dir):
    """Download required NLTK data to the specified directory"""
    # Set NLTK data path to output directory
    nltk_data_dir = os.path.join(output_dir, 'nltk_data')
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
        print(f"✅ NLTK data downloaded to {nltk_data_dir}")

def analyze_text_column(df, column_name):
    """Analyze text statistics for a column"""
    if column_name not in df.columns:
        return None

    # Remove null values for analysis
    text_data = df[column_name].dropna()

    if len(text_data) == 0:
        return None

    # Calculate statistics
    lengths = text_data.str.len()
    word_counts = text_data.str.split().str.len()
    all_words = ' '.join(text_data).split()

    stats = {
        'avg_length': float(lengths.mean()),
        'min_length': int(lengths.min()),
        'max_length': int(lengths.max()),
        'total_words': len(all_words),
        'unique_words': len(set(all_words))
    }

    return stats

def create_eda_plots(df, output_dir="outputs/001_data_analysis"):
    """Create EDA visualizations"""
    os.makedirs(output_dir, exist_ok=True)

    target_columns = [
        'prompt', 'Human_story', 'gemma-2-9b', 'mistral-7B',
        'qwen-2-72B', 'llama-8B', 'accounts/yi-01-ai/models/yi-large', 'GPT_4-o'
    ]

    # Text length distribution
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(target_columns, 1):
        if col in df.columns:
            plt.subplot(2, 4, i)
            lengths = df[col].dropna().str.len()
            plt.hist(lengths, bins=30, alpha=0.7)
            plt.title(f'{col}\nText Length Distribution')
            plt.xlabel('Text Length')
            plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/text_length_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Missing values visualization
    plt.figure(figsize=(10, 6))
    missing_data = df.isnull().sum()
    missing_data.plot(kind='bar')
    plt.title('Missing Values by Column')
    plt.ylabel('Number of Missing Values')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/missing_values.png', dpi=300, bbox_inches='tight')
    plt.close()

    return [
        f'{output_dir}/text_length_distributions.png',
        f'{output_dir}/missing_values.png'
    ]

def main():
    print("🚀 Starting Data Analysis...")

    # Initialize database
    db = CentralizedDB()

    # Output directory
    output_dir = "outputs/001_data_analysis"
    os.makedirs(output_dir, exist_ok=True)

    # Download NLTK data
    print("📥 Setting up NLTK data...")
    download_nltk_data(output_dir)

    try:
        # Load dataset
        print("📥 Loading dataset from HuggingFace...")
        dataset = load_dataset("gsingh1-py/train")
        df = dataset['train'].to_pandas()

        print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

        # Save dataset info to database
        db.save_dataset_info(df)

        # Define target columns
        target_columns = [
            'prompt', 'Human_story', 'gemma-2-9b', 'mistral-7B',
            'qwen-2-72B', 'llama-8B', 'accounts/yi-01-ai/models/yi-large', 'GPT_4-o'
        ]

        print(f"\n📊 Analyzing text statistics for {len(target_columns)} columns...")

        # Analyze each column and save statistics
        for col in target_columns:
            if col in df.columns:
                stats = analyze_text_column(df, col)
                if stats:
                    db.save_text_statistics(col, stats)
                    print(f"✅ {col}: avg_len={stats['avg_length']:.1f}, unique_words={stats['unique_words']}")
                else:
                    print(f"⚠️  {col}: No data to analyze")
            else:
                print(f"❌ {col}: Column not found")

        # Create EDA plots
        print("\n📈 Creating EDA visualizations...")
        plot_files = create_eda_plots(df)

        # Save dataset locally for other scripts
        dataset_path = f"{output_dir}/dataset.csv"
        df.to_csv(dataset_path, index=False)

        # Log successful execution
        db.log_script_execution(
            script_name="data_analysis.py",
            status="SUCCESS",
            description=f"Analyzed dataset with {df.shape[0]} rows and {df.shape[1]} columns",
            output_files=plot_files + [dataset_path]
        )

        print(f"\n✅ Data analysis completed successfully!")
        print(f"📁 Outputs saved to: outputs/data_analysis/")
        print(f"💾 Dataset saved to: {dataset_path}")
        print(f"🗄️  Metadata saved to: outputs/centralized.db")

    except Exception as e:
        db.log_script_execution(
            script_name="data_analysis.py",
            status="ERROR",
            description=f"Error: {str(e)}"
        )
        print(f"❌ Error: {e}")
        return False

    return True

if __name__ == "__main__":
    main()