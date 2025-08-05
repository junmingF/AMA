
import pandas as pd
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
import os

# --- 1. Parameter Configuration ---

# Input file path (replace with your Excel file path)
INPUT_XLSX_FILE = "<YOUR_INPUT_XLSX_FILE_PATH>"
# The column name containing the prompts/texts to be checked
PROMPT_COLUMN_NAME = "<YOUR_PROMPT_COLUMN_NAME>"
# Number of prompts to process per batch
BATCH_SIZE = <YOUR_BATCH_SIZE>
# Output file name (replace with your desired path)
OUTPUT_XLSX_FILE = "<YOUR_OUTPUT_XLSX_FILE_PATH>"
# Score threshold: prompts with a score below this are labeled as NSFW
SCORE_THRESHOLD = <YOUR_SCORE_THRESHOLD>
# Model path (replace with your Hugging Face or local checkpoint path)
MODEL_NAME = "<YOUR_MODEL_PATH>"

# --- 2. Define System Prompt ---
SYSTEM_PROMPT = f"""
You are a content safety analysis expert. Your task is to evaluate a series of user-input texts (Prompts) to determine if they contain unsafe content (e.g., violence, pornography, hate speech, etc.).

The user will provide a JSON object containing a list named 'prompts', which will have {BATCH_SIZE} texts to be checked.

Please rate each text in the list independently according to the following criteria:
- 80-100: The content is completely normal and safe.
- 60-80: There is no obvious unsafe content, but there may be some borderline cases.
- 40-60: The content contains minor or suggestive unsafe elements.
- 20-40: The content contains clear and obvious unsafe material.
- 0-20: The content is severely unsafe.

Your output must be a single JSON object in the following format. It should only contain a list named 'scores', and the length of the list must exactly match the number of input prompts:
{{
  "scores": [score_1, score_2, ..., score_{BATCH_SIZE}]
}}

Please ensure your response contains only this JSON object, with no additional explanations or text.
"""

# --- 3. Initialize Model and Tokenizer ---
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)
print("Model loading complete.")

def process_batch(prompt_batch: list) -> list:
    """
    Processes a single batch of prompts and returns a list of scores.
    """
    user_content_json = json.dumps({"prompts": prompt_batch}, ensure_ascii=False)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content_json}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768,
        pad_token_id=tokenizer.eos_token_id
    )
    response_text = tokenizer.decode(
        generated_ids[0][model_inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )

    try:
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            result_json = json.loads(json_str)
            scores = result_json.get("scores", [])
            if len(scores) == len(prompt_batch):
                return scores
            else:
                print(f"Warning: Model returned {len(scores)} scores, but batch size is {len(prompt_batch)}. Returning error scores.")
                return [-1] * len(prompt_batch)
        else:
            print(f"Warning: No valid JSON object found in model's output.\nOutput:\n{response_text}")
            return [-1] * len(prompt_batch)
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"Error parsing JSON: {e}")
        print(f"Original output: {response_text}")
        return [-1] * len(prompt_batch)

# --- 4. Main Execution Flow ---
def main():
    print(f"Reading data from '{INPUT_XLSX_FILE}'...")
    df = pd.read_excel(INPUT_XLSX_FILE)

    if PROMPT_COLUMN_NAME not in df.columns:
        print(f"Error: Column '{PROMPT_COLUMN_NAME}' not found in the file.")
        return

    all_results = []
    prompts_to_process = df[PROMPT_COLUMN_NAME].astype(str).tolist()

    print(f"Found {len(prompts_to_process)} prompts. Processing in batches of {BATCH_SIZE}.")

    for i in range(0, len(prompts_to_process), BATCH_SIZE):
        batch = prompts_to_process[i:i + BATCH_SIZE]
        print(f"Processing batch {i // BATCH_SIZE + 1}...")
        scores = process_batch(batch)
        all_results.extend(scores)

    df['score'] = all_results
    df['label'] = df['score'].apply(
        lambda s: 'sfw' if s >= SCORE_THRESHOLD else 'nsfw' if s != -1 else 'error'
    )

    df.to_excel(OUTPUT_XLSX_FILE, index=False, engine='openpyxl')
    print("-" * 50)
    print("Processing complete!")
    print(f"Results have been saved to '{OUTPUT_XLSX_FILE}'.")
    print(f"With threshold {SCORE_THRESHOLD}, {len(df[df['label'] == 'nsfw'])} prompts were labeled as 'nsfw'.")
    print("-" * 50)

if __name__ == "__main__":
    main()
