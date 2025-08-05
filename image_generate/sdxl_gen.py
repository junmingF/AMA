import argparse
import os
import sys
import torch
import pandas as pd
from diffusers import StableDiffusionXLPipeline

def parse_args():
    """Parse command-line arguments; this version only includes options required for the base model."""
    parser = argparse.ArgumentParser(
        description="Batch-generate images using only the SDXL base model based on prompts in an XLSX file."
    )
    # --- File and path arguments ---
    parser.add_argument(
        "--xlsx_file", type=str, default='/disk1/fjm/LLM/0713_with_sentence-thking.xlsx',
        help="Path to the XLSX file containing generation tasks (must include header)."
    )
    parser.add_argument(
        "--base_model_path", type=str, default="/disk1/fujm/stable-diffusion-xl-base-1.0",
        help="Local path or HuggingFace repo name for the SDXL base model."
    )
    parser.add_argument(
        "--output_folder", type=str, default="/disk1/fjm/base_ds/sdxl/mis",
        help="Directory to save the generated images."
    )
    # --- XLSX column definitions ---
    parser.add_argument(
        "--prompt_col_name", type=str, default="sentence",
        help="Prompt column name in the XLSX file."
    )
    parser.add_argument(
        "--output_name_col", type=str, default="image_name",
        help="Column name for the output file name."
    )
    # --- Batch processing arguments ---
    parser.add_argument(
        "--start_row", type=int, default=1,
        help="Start processing from this row number (1-based, excluding header)."
    )
    parser.add_argument(
        "--end_row", type=int, default=None,
        help="End processing at this row number (1-based, excluding header). Defaults to last row."
    )
    # --- Model & performance arguments ---
    parser.add_argument(
        "--cuda_device", type=int, default=1,
        help="CUDA device index to use."
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=7.5,
        help="Guidance scale (CFG scale), controls image-prompt alignment."
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=40,
        help="Number of inference steps."
    )
    parser.add_argument(
        "--num_outputs", type=int, default=1,
        help="Number of images to generate per prompt."
    )
    # --- Seed arguments ---
    parser.add_argument(
        "--seed", type=int, default=9,
        help="Fixed random seed, for reproducibility."
    )
    parser.add_argument(
        "--randomize_seed", action="store_true",
        help="Use different random seed for each task."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    # 1. Set device and dtype
    device = f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("⚠️ Warning: CUDA not detected, running on CPU.")
    torch_dtype = torch.float16 if device != "cpu" else torch.float32

    # 2. Load base model
    print(f"Loading SDXL base model: {args.base_model_path}")
    try:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            args.base_model_path,
            torch_dtype=torch_dtype,
            variant="fp16",
            use_safetensors=True
        ).to(device)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)

    # 3. Load XLSX (header=0 to use column names)
    try:
        df = pd.read_excel(args.xlsx_file, header=0, engine="openpyxl")
    except Exception as e:
        print(f"❌ Failed to read XLSX file {args.xlsx_file}: {e}")
        sys.exit(1)

    # 4. Determine processing range
    total_rows = len(df)
    start_index = max(0, args.start_row - 1)
    end_index = args.end_row if args.end_row and args.end_row <= total_rows else total_rows
    subset_df = df.iloc[start_index:end_index]
    print(f"\nReady to process tasks from row {start_index + 1} to {end_index}, total {len(subset_df)} tasks.")

    # 5. Process each task
    for idx, row in subset_df.iterrows():
        # Actual Excel row number: index + 2 (1 for header, 1 for 0-based index)
        current_row_num = idx + 2
        
        try:
            prompt = str(row[args.prompt_col_name]).strip()
            output_base_name = str(row[args.output_name_col]).strip()
        except KeyError as e:
            print(f"⏭️ Skipped row {current_row_num}: Column name {e} not found. Please check --prompt_col_name and --output_name_col arguments.")
            print(f"    Available column names: {list(df.columns)}")
            continue

        if not prompt or pd.isna(prompt) or not output_base_name or pd.isna(output_base_name):
            print(f"⏭️ Skipped row {current_row_num}: Missing prompt or output file name.")
            continue

        # 6. Set random seed
        seed = torch.seed() if args.randomize_seed else args.seed
        generator = torch.Generator(device=device).manual_seed(seed)

        # 7. Inference (single step, direct image)
        print(f"▶️ Generating for row {current_row_num}: '{prompt[:80]}...'")
        try:
            images = pipe(
                prompt=prompt,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                num_images_per_prompt=args.num_outputs,
                generator=generator
            ).images
        except Exception as e:
            print(f"❌ Inference failed for row {current_row_num}: {e}")
            continue

        # 8. Save images
        base_name, _ = os.path.splitext(output_base_name)
        for i, image_out in enumerate(images, start=1):
            safe_base_name = base_name.replace("/", "_").replace("\\", "_")
            save_name = f"{safe_base_name}_{i}.png" if args.num_outputs > 1 else f"{safe_base_name}.png"
            save_path = os.path.join(args.output_folder, save_name)
            image_out.save(save_path)
        
        print(f"✅ Successfully generated {len(images)} image(s) for row {current_row_num}.")

if __name__ == "__main__":
    main()