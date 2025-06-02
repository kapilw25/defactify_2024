import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score

# Load the dataset
df = pd.read_csv('results/baseline_benchmark_33_results_merged_with_groundTruth_modified.csv')

print("Dataset shape:", df.shape)

# Create a binary classification problem:
# 1 if the predicted model matches the ground truth, 0 otherwise
y_true = [1] * len(df)  # Ideally, all predictions should match ground truth
y_pred = (df['best_LLM_model'] == df['Label_B']).astype(int)

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# Print results
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Count matches and mismatches
matches = (df['best_LLM_model'] == df['Label_B']).sum()
total = len(df)
print(f"\nMatches: {matches} out of {total} ({matches/total:.4f})")

# For human stories (Label_A == 0)
human_stories = df[df['Label_A'] == 0]
human_matches = (human_stories['best_LLM_model'] == human_stories['Label_B']).sum()
print(f"\nHuman stories (Label_A == 0):")
print(f"Matches: {human_matches} out of {len(human_stories)} ({human_matches/len(human_stories):.4f})")

# For AI-generated content (Label_A == 1)
ai_content = df[df['Label_A'] == 1]
ai_matches = (ai_content['best_LLM_model'] == ai_content['Label_B']).sum()
print(f"\nAI-generated content (Label_A == 1):")
print(f"Matches: {ai_matches} out of {len(ai_content)} ({ai_matches/len(ai_content):.4f})")
