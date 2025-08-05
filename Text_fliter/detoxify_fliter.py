
import argparse
import os

import pandas as pd
from detoxify import Detoxify


def load_input_text(input_obj, column_name=None):
    """
    Load input text(s).
    - If input_obj is a string, returns it directly.
    - If input_obj is a .txt file path, returns the lines from the file.
    - If input_obj is a .xlsx file path, returns the specified column as a list.
    """
    if isinstance(input_obj, str) and os.path.isfile(input_obj):
        # Input is a file path
        if input_obj.endswith(".txt"):
            print("Loading text from .txt file...")
            with open(input_obj, 'r', encoding='utf-8') as f:
                text = f.read().splitlines()
        elif input_obj.endswith(".xlsx"):
            print(f"Loading text from column '{column_name}' in .xlsx file '{input_obj}'...")
            if column_name is None:
                raise ValueError("Error: When using a .xlsx file, you must specify --column_name.")
            try:
                df = pd.read_excel(input_obj)
                if column_name not in df.columns:
                    raise ValueError(f"Error: Column '{column_name}' not found. Available columns: {list(df.columns)}")
                text = df[column_name].dropna().astype(str).tolist()
            except Exception as e:
                raise IOError(f"Failed to read Excel file: {e}")
        else:
            raise ValueError("Invalid file type: only .txt or .xlsx files are supported.")
    elif isinstance(input_obj, str):
        # Input is a plain string
        text = input_obj
    else:
        raise ValueError("Invalid input type: input must be a string, .txt path, or .xlsx path.")
    
    if not text:
        raise ValueError("No text was loaded: check your file, column, or input value.")
        
    return text


def run(model_name, input_obj, dest_file, from_ckpt, column_name=None, device="cpu"):
    """
    Load the model and perform inference on input text.
    Show results as a pandas DataFrame, optionally save to file.
    """
    text = load_input_text(input_obj, column_name)
    
    if model_name is not None:
        model = Detoxify(model_name, device=device)
    else:
        model = Detoxify(checkpoint=from_ckpt, device=device)
    
    print("Model loaded. Starting prediction...")
    res = model.predict(text)

    res_df = pd.DataFrame(
        res,
        index=[text] if isinstance(text, str) else text,
    ).round(5)
    
    print("\n--- Prediction Results ---")
    print(res_df)
    
    if dest_file is not None:
        res_df.index.name = "input_text"
        res_df.to_csv(dest_file)
        print(f"\nResults saved to: {dest_file}")

    return res


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Analyze toxicity using Detoxify, supporting text string, .txt file, or .xlsx file input.")
    parser.add_argument(
        "--input",
        type=str,
        default='/disk1/fjm/detoxify/afterLG/sneakynsfw_SFW_output_sfw.xlsx',
        help="Text for analysis, path to a .txt file, or path to a .xlsx file.",
    )
    parser.add_argument(
        "--column_name",
        type=str,
        default='prompt',
        help="Column name to use when the input is a .xlsx file.",
    )
    parser.add_argument(
        "--model_name",
        default="unbiased",
        type=str,
        choices=["original", "unbiased", "multilingual"],
        help="Torch.hub model name to use (default: unbiased)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="Device to load the model on (e.g., 'cpu', 'cuda', 'cuda:0')",
    )
    parser.add_argument(
        "--from_ckpt_path",
        default=None,
        type=str,
        help="Optional local checkpoint path to load the model from (default: None)",
    )
    parser.add_argument(
        "--save_to",
        default='/disk1/fjm/detoxify/afterDE/sneaky.csv',
        type=str,
        help="Path to save the model results as a CSV (e.g., results.csv)",
    )

    args = parser.parse_args()

    # --- Argument validation ---
    if args.from_ckpt_path and args.model_name != "unbiased":
        import sys
        if any(f'--model_name={arg}' in sys.argv for arg in ["original", "multilingual"]):
            raise ValueError("Please specify only one model source: --from_ckpt_path or --model_name.")
    if args.from_ckpt_path:
        assert os.path.isfile(args.from_ckpt_path), f"Checkpoint file not found: {args.from_ckpt_path}"
        args.model_name = None

    run(
        model_name=args.model_name,
        input_obj=args.input,
        dest_file=args.save_to,
        from_ckpt=args.from_ckpt_path,
        column_name=args.column_name,
        device=args.device,
    )
