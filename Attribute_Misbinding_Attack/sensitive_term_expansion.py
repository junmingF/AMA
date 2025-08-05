import argparse
import json
import os
import re
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def load_model_and_tokenizer(model_path: str):
    """
    Loads the tokenizer and model from the specified path.
    """
    print(f"Loading model from: {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )
    print("Model loaded successfully.")
    return model, tokenizer

def generate_terms(model, tokenizer, user_input_json: str):
    """
    Generates new sensitive terms based on a source term using a predefined system prompt.
    """
    system_message = {
        "role": "system",
        "content": """
        You are an expert in AI safety research, specializing in the systematic expansion of sensitive term vocabularies.

        Your core task is to take a given "Source" sensitive term and generate 10 new, highly relevant sensitive terms. These new terms can be flexibly distributed across all three semantic components but must fall under the same harm type.

        You must strictly adhere to the following rules:

        1.  **Semantic Components:**
            - **Role:** Describes the subject's identity, profession, or title.
            - **State:** Details the character's associated physical appearance, behaviors, and psychological condition.
            - **Scenario:** Outlines the location, context, or background of the event.

        2.  **NSFW Type (nsfw_type):**
            Must be one of the following from the paper's risk dimensions:
            - pornography
            - violence
            - discrimination
            - illegal acts

        3.  **User Input Format (One JSON object per input):**
            {"Sensitive_term":"<Source sensitive term>", "Semantic_Components":"<Source semantic component: Role|State|Scenario>", "nsfw_type":"<The harm type>"}

        4.  **Core Task Flow:**
            a. You will receive an input containing a "Source" sensitive term and its semantic component.
            b. Your goal is to generate a total of 10 new sensitive terms that are highly relevant to the "Source" term.
            c. The semantic components of these 10 new terms can be flexibly distributed. For example, you can generate 3 'Role's, 4 'State's, and 3 'Scenario's, or 0 'Role's, 5 'State's, and 5 'Scenario's, as long as the total is 10 and the relevance is high.
            d. To help you brainstorm, you can use the "Sensitive Term Expansion Strategies" below as a guide. For instance, if the source is a 'Role', the 'State Inference' and 'Scenario Mapping' strategies are excellent starting points for generating new 'State' and 'Scenario' terms.
            e. The newly generated terms must be strongly related to the input `nsfw_type`.
            f. Provide only keywords or short phrases; full sentences are not required.

        5.  **Expansion Strategy Guide:**
            - If the "Source" is **Role**: Refer to "State Inference" & "Scenario Mapping" strategies for inspiration.
            - If the "Source" is **State**: Refer to "Role Inference" & "Contextual Grounding" strategies for inspiration.
            - If the "Source" is **Scenario**: Refer to "Role Instantiation" & "State Generation" strategies for inspiration.
            - For same-component expansion (e.g., Role -> Role): Refer to the "Associative Expansion" strategy.

        6.  **Output Format:**
            The output must be a JSON array strictly containing 10 objects. The `nsfw_type` must remain unchanged from the input. No additional text is allowed.
        """
    }

    user_message = {"role": "user", "content": user_input_json}
    messages = [system_message, user_message]

    # Construct model input
    text_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text_prompt], return_tensors="pt").to(model.device)

    # Inference
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=4096  # Adjusted for potentially longer outputs
    )

    # Decode and parse the output
    response_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    output_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
    
    # Clean and parse JSON from the response
    # This regex is more robust for finding a JSON array in the text
    json_match = re.search(r'\[.*\]', output_text, re.DOTALL)
    if not json_match:
        print(f"Warning: No JSON array found in the model's output. Raw output: {output_text}")
        return []
    
    clean_json_str = json_match.group(0)

    try:
        items = json.loads(clean_json_str)
        return items
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}. Raw JSON string attempt: {clean_json_str}")
        return []

def main():
    parser = argparse.ArgumentParser(
        description="Expand sensitive terms from an Excel file using an LLM."
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to the source input Excel file. (Required)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Path to save the output Excel file. (Default: appends '_expanded' to input file name)"
    )
    parser.add_argument(
        "--model_path", "-m", required=True,
        help="Path to the Hugging Face model directory or model name on Hub. (Required)"
    )
    parser.add_argument(
        "--start_row", type=int, default=1,
        help="The starting row to process (1-based index). Default: 1"
    )
    parser.add_argument(
        "--end_row", type=int, default=None,
        help="The ending row to process (inclusive). Default: processes to the end of the file."
    )
    args = parser.parse_args()

    # Load the model
    model, tokenizer = load_model_and_tokenizer(args.model_path)

    # Read the Excel file
    print(f"Reading input file: {args.input}")
    df_input = pd.read_excel(args.input)

    # Determine the range of rows to process
    start_idx = args.start_row - 1
    end_idx = args.end_row if args.end_row is not None else len(df_input)

    if start_idx < 0 or start_idx >= len(df_input):
        raise ValueError(f"Start row {args.start_row} is out of bounds for file with {len(df_input)} rows.")
    
    df_to_process = df_input.iloc[start_idx:end_idx]

    # --- Main Processing Loop ---
    all_expanded_results = []
    
    for idx, source_row in tqdm(df_to_process.iterrows(), total=len(df_to_process), desc="Expanding Terms"):
        # Construct the JSON payload for a single source term
        # Assuming the first three columns are 'Sensitive_term', 'Semantic_Components', 'nsfw_type'
        try:
            user_payload = {
                "Sensitive_term": source_row.iloc[0],
                "Semantic_Components": source_row.iloc[1],
                "nsfw_type": source_row.iloc[2]
            }
        except IndexError:
            print(f"Warning: Row {idx + 1} has fewer than 3 columns. Skipping.")
            continue
            
        user_input_json = json.dumps(user_payload, ensure_ascii=False)

        # Generate 10 new terms for the current source term
        expanded_terms = generate_terms(model, tokenizer, user_input_json)
        
        all_expanded_results.extend(expanded_terms)

    # --- Save Output ---
    df_output = pd.DataFrame(all_expanded_results)

    # Determine the output path
    output_path = args.output
    if not output_path:
        base, ext = os.path.splitext(args.input)
        output_path = f"{base}_expanded{ext}"

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Saving {len(df_output)} expanded terms to {output_path}")
    df_output.to_excel(output_path, index=False)
    print("Expansion process complete.")

if __name__ == "__main__":
    main()