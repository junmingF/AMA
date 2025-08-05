import argparse
import os
import sys
import torch
import pandas as pd
from diffusers import FluxPipeline

def parse_args():
    """Parse command-line arguments, customized for the FLUX.1-dev model."""
    parser = argparse.ArgumentParser(
        description="Batch generate images using the FLUX.1-dev model based on prompts from an XLSX file."
    )
    # --- File and Path Arguments ---
    parser.add_argument(
        "--xlsx_file", type=str, default='/disk1/fjm/LLM/0713_with_sentence-thking.xlsx',
        help="Path to the XLSX file containing generation tasks (must include a header row)."
    )
    parser.add_argument(
        "--model_path", type=str, default="/disk1/fujm/FLUX.1-dev",
        help="Local path or HuggingFace repository name of the FLUX.1-dev model."
    )
    parser.add_argument(
        "--output_folder", type=str, default="/disk1/fjm/base_ds/flux_dev/mis",
        help="Directory to save the generated images."
    )
    # --- XLSX Column Names (MODIFIED: index to name) ---
    parser.add_argument(
        "--prompt_col_name", type=str, default="sentence",
        help="The column name for prompts in the XLSX file."
    )
    parser.add_argument(
        "--output_name_col", type=str, default="image_name",
        help="The column name used as the output file name."
    )
    # --- Batch Processing Arguments ---
    parser.add_argument(
        "--start_row", type=int, default=1,
        help="The starting data row number to process (1-based, excluding header)."
    )
    parser.add_argument(
        "--end_row", type=int, default=None,
        help="The ending data row number to process (1-based, excluding header), default is the last row."
    )
    # --- Model and Performance Arguments ---
    parser.add_argument(
        "--cuda_device", type=int, default=2,
        help="CUDA device index to use."
    )
    parser.add_argument(
        "--cpu_offload", action="store_true",
        help="Enable CPU offload to save GPU memory. Not needed if you have enough VRAM."
    )
    # --- FLUX Model Arguments ---
    parser.add_argument(
        "--width", type=int, default=512,
        help="Width of the generated image."
    )
    parser.add_argument(
        "--height", type=int, default=512,
        help="Height of the generated image."
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=3.5,
        help="Guidance scale (CFG Scale). Lower values are recommended for FLUX."
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=50,
        help="Number of inference steps."
    )
    parser.add_argument(
        "--max_sequence_length", type=int, default=512,
        help="Maximum sequence length for the text encoder."
    )
    parser.add_argument(
        "--num_outputs", type=int, default=1,
        help="Number of images to generate for each prompt."
    )
    # --- Seed Arguments ---
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Fixed random seed for reproducibility."
    )
    parser.add_argument(
        "--randomize_seed", action="store_true",
        help="Use a different random seed for each task."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    # 1. Set device and dtype
    if torch.cuda.is_available():
        device = f"cuda:{args.cuda_device}"
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        print("⚠️ Warning: CUDA not detected. Running on CPU.")
        device = "cpu"
        torch_dtype = torch.float32

    # 2. Load model
    print(f"Loading FLUX.1-dev model: {args.model_path}")
    try:
        pipe = FluxPipeline.from_pretrained(
            args.model_path,
            torch_dtype=torch_dtype
        )
        # Enable CPU offload if needed
        if args.cpu_offload and device != "cpu":
            print("...Enabling model CPU offload to save GPU memory.")
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)

    # 3. Read XLSX file (MODIFIED: header=0 to use column names)
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
        # Actual Excel row number = pandas index + 2 (1 for header, 1 for 0-based index)
        current_row_num = idx + 2
        
        try:
            # MODIFIED: Get data by column name
            prompt = str(row[args.prompt_col_name]).strip()
            output_base_name = str(row[args.output_name_col]).strip()
        except KeyError as e:
            print(f"⏭️ Skipping row {current_row_num}: column name {e} does not exist. Please check --prompt_col_name and --output_name_col.")
            print(f"    Available columns: {list(df.columns)}")
            continue

        if not prompt or pd.isna(prompt) or not output_base_name or pd.isna(output_base_name):
            print(f"⏭️ Skipping row {current_row_num}: Missing prompt or output file name.")
            continue

        # 6. Set random seed
        seed = torch.seed() if args.randomize_seed else args.seed
        # For FLUX: If CPU offload is enabled, the generator should be placed on CPU
        generator_device = "cpu" if args.cpu_offload and device != "cpu" else device
        generator = torch.Generator(device=generator_device).manual_seed(seed)

        # 7. Inference
        print(f"▶️ Generating for row {current_row_num}: '{prompt[:80]}...'")
        try:
            images = pipe(
                prompt=prompt,
                width=args.width,
                height=args.height,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                max_sequence_length=args.max_sequence_length,
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
        
        print(f"✅ Successfully generated {len(images)} images for row {current_row_num}.")

if __name__ == "__main__":
    main()