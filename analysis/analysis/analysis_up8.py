import sys
sys.path.append('..')
import importlib
import experiment_data as ed
import pandas as pd
import google.generativeai as genai
import os
import time
from typing import List
importlib.reload(ed)  # Force reload to get latest version

# Get the experiment object from main function
experiment = ed.main()

# Use the to_dataframe_contributions method
df = experiment.to_dataframe_contributions()

# Create DataFrame of all chat messages using the new method
chat_df = experiment.to_dataframe_chat()

# Promise examples from the RTF file
PROMISE_EXAMPLES = [
    "ok im in",
    "i'm down",
    "yeah thats fine with me",
    "good with me",
    "yeah lets do 25",
    "lets do it",
    "25 we go again",
    "I'm still doing 25 lol",
    "deal",
    "lets all do 25",
    "ya lets all donate everything possible",
    "lets keep it this way",
    "yeah do it everytime",
    "everyone do all 25",
    "done",
    "yes",
    "yup",
    "yea",
    "lets put 25 in the group account so that we all can earn 4",
    "agreed"
]

def setup_gemini_api():
    """Setup Gemini API configuration."""
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash-lite')
    return model

def classify_promise(text: str, model, examples: List[str]) -> int:
    """Classify if a text is a promise using Gemini API.
    
    Args:
        text: The text to classify
        model: Gemini model instance
        examples: List of promise examples
    
    Returns:
        1 if promise, 0 if not a promise
    """
    examples_text = '\n'.join([f'- "{example}"' for example in examples])
    
    prompt = f"""Classify this text as a promise or not a promise. Return 1 if it is a promise and 0 if it is not. 

Here are some examples of promises:
{examples_text}

Text to classify: "{text}"

Return only the number (1 or 0):"""
    
    try:
        response = model.generate_content(prompt)
        result = response.text.strip()
        
        # Parse the response to get 1 or 0
        if '1' in result:
            return 1
        elif '0' in result:
            return 0
        else:
            # If unclear response, default to 0
            print(f"Unclear response for '{text}': {result}")
            return 0
            
    except Exception as e:
        print(f"Error classifying '{text}': {e}")
        return 0  # Default to not a promise on error

def classify_messages_batch(messages: List[str], model, examples: List[str]) -> List[int]:
    """Classify messages in batches.
    
    Args:
        messages: List of message texts to classify
        model: Gemini model instance
        examples: List of promise examples
    
    Returns:
        List of classifications (1 for promise, 0 for not promise)
    """
    results = []
    
    for i, message in enumerate(messages):
        result = classify_promise(message, model, examples)
        results.append(result)
        
        if i % 100 == 0:
            print(f"Processed {i+1}/{len(messages)} messages")
    
    return results

# Setup Gemini API
print("Setting up Gemini API...")
model = setup_gemini_api()

# Classify all chat messages as promises or not
if chat_df is not None and len(chat_df) > 0:
    print(f"Classifying {len(chat_df)} chat messages for promise content...")
    
    # Get list of messages
    messages = chat_df['message'].tolist()
    
    # Classify messages
    promise_classifications = classify_messages_batch(messages, model, PROMISE_EXAMPLES)
    
    # Add promise classification to dataframe
    chat_df['is_promise'] = promise_classifications
    
    print(f"Promise classification complete!")
    print(f"Found {sum(promise_classifications)} promises out of {len(messages)} messages")
    print(f"Promise rate: {sum(promise_classifications)/len(messages)*100:.1f}%")
    
    # Output validation samples
    print("\n" + "="*80)
    print("VALIDATION SAMPLES")
    print("="*80)
    
    # Get promises and non-promises
    promises_df = chat_df[chat_df['is_promise'] == 1]
    non_promises_df = chat_df[chat_df['is_promise'] == 0]
    
    print(f"\n🟢 CLASSIFIED AS PROMISES ({len(promises_df)} total):")
    print("-" * 50)
    
    # Show up to 50 promises
    promise_sample = promises_df['message'].head(50).tolist()
    for i, promise in enumerate(promise_sample, 1):
        print(f"{i:2}. \"{promise}\"")
    
    print(f"\n🔴 CLASSIFIED AS NON-PROMISES ({len(non_promises_df)} total):")
    print("-" * 50)
    
    # Show up to 50 non-promises
    non_promise_sample = non_promises_df['message'].head(50).tolist()
    for i, non_promise in enumerate(non_promise_sample, 1):
        print(f"{i:2}. \"{non_promise}\"")
    
    print("\n" + "="*80)
    
    # Save the updated dataframe to results folder
    results_dir = "../results"
    os.makedirs(results_dir, exist_ok=True)
    
    output_file = os.path.join(results_dir, "chat_with_promises.csv")
    chat_df.to_csv(output_file, index=False)
    
    print(f"\n💾 Saved chat data with promise classifications to: {output_file}")
    print(f"Columns: {list(chat_df.columns)}")
    print(f"Shape: {chat_df.shape}")
    
else:
    print("No chat data found to classify")












