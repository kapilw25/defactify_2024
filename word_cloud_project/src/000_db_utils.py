import sqlite3
import pandas as pd
import json
import os
from datetime import datetime

class CentralizedDB:
    def __init__(self, db_path="outputs/centralized.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.init_database()

    def init_database(self):
        """Initialize the centralized database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create metadata table to track script executions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS script_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                script_name TEXT NOT NULL,
                execution_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT,
                description TEXT,
                output_files TEXT
            )
        """)

        # Create dataset_info table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dataset_info (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                total_rows INTEGER,
                total_columns INTEGER,
                column_names TEXT,
                missing_values TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create 001_text_statistics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS t001_text_statistics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                column_name TEXT,
                avg_length REAL,
                min_length INTEGER,
                max_length INTEGER,
                total_words INTEGER,
                unique_words INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create 002_word_cloud_metadata table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS t002_word_cloud_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                column_name TEXT,
                image_path TEXT,
                top_words TEXT,
                word_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create 003_comparative_analysis table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS t003_comparative_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                column_name TEXT,
                real_sample TEXT,
                fake_sample TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create 004_sentiment_analysis table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS t004_sentiment_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                column_name TEXT,
                avg_sentiment REAL,
                sentiment_distribution TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()
        print(f"✅ Database initialized at {self.db_path}")

    def log_script_execution(self, script_name, status, description="", output_files=[]):
        """Log script execution details"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO script_metadata (script_name, status, description, output_files)
            VALUES (?, ?, ?, ?)
        """, (script_name, status, description, json.dumps(output_files)))

        conn.commit()
        conn.close()
        print(f"✅ Logged execution: {script_name} - {status}")

    def save_dataset_info(self, df):
        """Save dataset information"""
        conn = sqlite3.connect(self.db_path)

        missing_info = df.isnull().sum().to_dict()

        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO dataset_info (total_rows, total_columns, column_names, missing_values)
            VALUES (?, ?, ?, ?)
        """, (
            len(df),
            len(df.columns),
            json.dumps(list(df.columns)),
            json.dumps(missing_info)
        ))

        conn.commit()
        conn.close()
        print(f"✅ Dataset info saved: {len(df)} rows, {len(df.columns)} columns")

    def save_text_statistics(self, column_name, stats):
        """Save text statistics for a column"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO t001_text_statistics
            (column_name, avg_length, min_length, max_length, total_words, unique_words)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            column_name,
            stats['avg_length'],
            stats['min_length'],
            stats['max_length'],
            stats['total_words'],
            stats['unique_words']
        ))

        conn.commit()
        conn.close()
        print(f"✅ Text statistics saved for {column_name}")

    def save_wordcloud_metadata(self, column_name, image_path, top_words, word_count):
        """Save word cloud metadata"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO t002_word_cloud_metadata
            (column_name, image_path, top_words, word_count)
            VALUES (?, ?, ?, ?)
        """, (column_name, image_path, json.dumps(top_words), word_count))

        conn.commit()
        conn.close()
        print(f"✅ Word cloud metadata saved for {column_name}")

    def get_dataset_info(self):
        """Retrieve latest dataset information"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("""
            SELECT * FROM dataset_info
            ORDER BY created_at DESC LIMIT 1
        """, conn)
        conn.close()
        return df

    def get_text_statistics(self):
        """Retrieve all text statistics"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM t001_text_statistics", conn)
        conn.close()
        return df

    def get_execution_log(self):
        """Retrieve script execution log"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("""
            SELECT * FROM script_metadata
            ORDER BY execution_time DESC
        """, conn)
        conn.close()
        return df

if __name__ == "__main__":
    # Test the database setup
    db = CentralizedDB()
    print("Database setup completed successfully!")