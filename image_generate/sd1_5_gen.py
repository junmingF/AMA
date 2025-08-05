import argparse
import os
import sys
import torch
import pandas as pd
from diffusers import StableDiffusionPipeline

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Batch-generate high-quality, print-ready images using StableDiffusionPipeline based on prompts from an XLSX file."
    )
    # --- File and path arguments ---
    parser.add_argument(
        "--xlsx_file", type=str, default='/disk1/fjm/LLM/0713_with_sentence-thking.xlsx',
        help="Path to the XLSX file containing generation tasks (must include header row)."
    )
    parser.add_argument(
        "--model_path", type=str, default="/disk1/fujm/stable-diffusion-v1-5",
        help="Local path or HuggingFace repo name for Stable Diffusion v1.5 model."
    )
    parser.add_argument(
        "--output_folder", type=str, default="/disk1/fjm/base_ds/sd1-5/mis",
        help="Directory to save the generated images."
    )
    # --- XLSX file content definition ---
    parser.add_argument(
        "--prompt_col_name", type=str, default="sentence",
        help="Name of the prompt column in the XLSX file."
    )
    parser.add_argument(
        "--output_name_col", type=str, default="image_name",
        help="Name of the output file name column in the XLSX file."
    )
    # --- Batch processing arguments ---
    parser.add_argument(
        "--start_row", type=int, default=1,
        help="Start row number for processing (1-based, excluding header)."
    )
    parser.add_argument(
        "--end_row", type=int, default=None,
        help="End row number for processing (1-based, excluding header). Defaults to the last row."
    )
    # --- Model and performance arguments ---
    parser.add_argument(
        "--cuda_device", type=int, default=0, help="CUDA device index to use."
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=7.5, help="Classifier-free guidance scale (CFG scale)."
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=50, help="Number of inference steps."
    )
    parser.add_argument(
        "--num_outputs", type=int, default=1, help="Number of images to generate per prompt."
    )
    # --- Resolution arguments ---
    parser.add_argument(
        "--width", type=int, default=512, help="Width of the generated images."
    )
    parser.add_argument(
        "--height", type=int, default=512, help="Height of the generated images."
    )
    # --- Seed arguments ---
    parser.add_argument(
        "--seed", type=int, default=9, help="Fixed random seed."
    )
    parser.add_argument(
        "--randomize_seed", action="store_true", help="Use a different random seed for each task."
    )
    # --- Prompt enhancement arguments ---
    parser.add_argument(
        "--quality_prompt", type=str, 
        default="masterpiece, best quality, high resolution, photorealistic, 8k, UHD, detailed, sharp focus, professional photography",
        help="Quality boost prompt appended to each input prompt."
    )
    parser.add_argument(
        "--negative_prompt", type=str,
        default="(worst quality, low quality:1.4), (blurry:1.2), ugly, deformed, disfigured, bad anatomy, extra limbs, fused fingers, too many fingers, malformed hands, text, watermark, signature, username, logo, jpeg artifacts, lowres",
        help="General negative prompt."
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    # 1. Select device
    device = f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Warning: CUDA not detected, running on CPU.")

    # 2. Select numeric precision
    torch_dtype = torch.float16 if device != "cpu" else torch.float32

    # 3. Load model
    print(f"Loading model: {args.model_path} to {device}, dtype={torch_dtype}")
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            args.model_path,
            torch_dtype=torch_dtype
        ).to(device)
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)

    # 4. Read XLSX file
    try:
        df = pd.read_excel(args.xlsx_file, header=0, engine="openpyxl")
    except Exception as e:
        print(f"Failed to read XLSX file {args.xlsx_file}: {e}")
        sys.exit(1)

    # 5. Determine processing range
    total_rows = len(df)
    start_index = max(0, args.start_row - 1)
    end_index = args.end_row if args.end_row and args.end_row <= total_rows else total_rows
    subset_df = df.iloc[start_index:end_index]

    print(f"Preparing to process tasks from row {start_index + 1} to {end_index}, total {len(subset_df)} tasks.")
    print(f"Print quality settings: Resolution={args.width}x{args.height}, CFG={args.guidance_scale}, Steps={args.num_inference_steps}")

    # 6. Process each row/task
    for idx, row in subset_df.iterrows():
        # current_row_num is the actual Excel row number (1-based header + 1-based index)
        current_row_num = idx + 2 
        
        try:
            prompt_base = str(row[args.prompt_col_name]).strip()
            output_base_name = str(row[args.output_name_col]).strip()
        except KeyError as e:
            print(f"[Row {current_row_num}] Skipped: column name {e} not found. Please check --prompt_col_name and --output_name_col arguments.")
            print(f"    Available column names: {list(df.columns)}")
            continue

        if not prompt_base or pd.isna(prompt_base) or not output_base_name or pd.isna(output_base_name):
            print(f"[Row {current_row_num}] Skipped: Missing prompt or output file name.")
            continue
            
        full_prompt = f"{prompt_base}, {args.quality_prompt}"

        # 7. Handle random seed
        seed = torch.seed() if args.randomize_seed else args.seed
        generator = torch.Generator(device=device).manual_seed(seed)

        # 8. Inference
        print(f"[Row {current_row_num}] Generating... Prompt: '{prompt_base[:80]}...'")
        try:
            outputs = pipe(
                prompt=full_prompt,
                negative_prompt=args.negative_prompt,
                height=args.height,
                width=args.width,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                num_images_per_prompt=args.num_outputs,
                generator=generator
            ).images
        except Exception as e:
            print(f"[Row {current_row_num}] Inference failed: {e}")
            continue

        # 9. Save images
        base_name, _ = os.path.splitext(output_base_name)
        for i, image_out in enumerate(outputs, start=1):
            safe_base_name = base_name.replace("/", "_").replace("\\", "_")
            save_name = f"{safe_base_name}_{i}.png" if args.num_outputs > 1 else f"{safe_base_name}.png"
            save_path = os.path.join(args.output_folder, save_name)
            image_out.save(save_path)
        
        print(f"[Row {current_row_num}] Successfully generated {len(outputs)} image(s), saved to {args.output_folder}")

if __name__ == "__main__":
    main()