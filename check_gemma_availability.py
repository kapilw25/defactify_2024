from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the API key from environment variables
hf_api_key = os.getenv("HF_API_KEY")

print(f"Using API key: {hf_api_key[:5]}...")

client = InferenceClient(
    provider="nebius",
    api_key=hf_api_key,
)

try:
    print("Attempting to connect to the model...")
    stream = client.chat.completions.create(
        model="google/gemma-3-27b-it",
        messages=[
            {
                "role": "user",
                "content": "What is the capital of France?"
            }
        ],
        temperature=0.5,
        top_p=0.7,
        stream=True,
    )

    print("Connection successful! Response:")
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="")
    print("\n")
    
except Exception as e:
    print(f"Error: {e}")
    print("\nPossible issues:")
    print("1. Your API key might be incorrect or expired")
    print("2. You might not have accepted the model's license agreement")
    print("3. The model might not be available through the specified provider")
    print("\nTo accept the model license, visit: https://huggingface.co/google/gemma-3-27b-it")
