import argparse
import os
import sys
import torch
import pandas as pd
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image

def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch-generate images using FluxKontextPipeline based on prompts in an XLSX file."
    )
    parser.add_argument(
        "--xlsx_file", type=str, default="/disk1/fjm/LLM/afterfliterprompt/i2p_sfw.xlsx",
        help="Path to the XLSX file (by default, no header). First column: image_name, second column: prompt."
    )
    parser.add_argument(
        "--model_path", type=str, default="/disk1/fujm/FLUX.1-Kontext-dev",
        help="Local path or HuggingFace repo path to FluxKontextPipeline model."
    )
    parser.add_argument(
        "--image_folder", type=str, default="/disk1/fjm/img",
        help="Folder containing input images."
    )
    parser.add_argument(
        "--output_folder", type=str, default="/disk1/fjm/resultimg/kontext/i2p",
        help="Directory to save generated images."
    )
    parser.add_argument(
        "--start_row", type=int, default=0,
        help="Starting row (1-based, inclusive). Default: 1"
    )
    parser.add_argument(
        "--end_row", type=int, default=1700,
        help="Ending row (1-based, inclusive). Default: last row"
    )
    parser.add_argument(
        "--cuda_device", type=int, default=2,
        help="CUDA device index."
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=2.5,
        help="Classifier-free guidance scale."
    )
    parser.add_argument(
        "--num_outputs", type=int, default=1,
        help="Number of images to generate for each input."
    )
    parser.add_argument(
        "--seed", type=int, default=9,
        help="Random seed (can specify, or use with --randomize_seed)."
    )
    parser.add_argument(
        "--randomize_seed", action="store_true",
        help="Randomize seed for each image."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    # Choose device
    if torch.cuda.is_available():
        torch.cuda.set_device(args.cuda_device)
        device = f"cuda:{args.cuda_device}"
    else:
        print("Warning: CUDA not detected. Running on CPU.")
        device = "cpu"

    # Select dtype
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device == "cpu":
        torch_dtype = torch.float32

    # Load model
    print(f"Loading model from {args.model_path} to {device}, dtype={torch_dtype}")
    pipe = FluxKontextPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype
    ).to(device)

    # Read XLSX file
    try:
        df = pd.read_excel(args.xlsx_file, engine="openpyxl")
    except Exception as e:
        print(f"Failed to read {args.xlsx_file}: {e}")
        sys.exit(1)

    total = len(df)
    start = max(0, args.start_row - 1)
    end = args.end_row if args.end_row and args.end_row <= total else total
    subset = df.iloc[start:end]

    for idx, row in subset.iterrows():
        row_no = idx + 1
        img_name = str(row['image_name']).strip()
        prompt = str(row['prompt']).strip()
        if not img_name or not prompt:
            print(f"[row {row_no}] Skipped: missing image_name or prompt.")
            continue

        # Handle seed
        seed = torch.seed() if args.randomize_seed else args.seed
        generator = torch.Generator(device=device).manual_seed(seed)

        # Prepare input image
        img_path = os.path.join(args.image_folder, img_name)
        if not os.path.isfile(img_path):
            print(f"[row {row_no}] Skipped: file not found {img_path}")
            continue
        pil_img = load_image(img_path)

        # Inference
        try:
            outputs = pipe(
                image=pil_img,
                prompt=prompt,
                guidance_scale=args.guidance_scale,
                num_images_per_prompt=args.num_outputs,
                generator=generator
            ).images
        except Exception as e:
            print(f"[row {row_no}] Inference failed: {e}")
            continue

        # Save images
        base, _ = os.path.splitext(img_name)
        for i, out in enumerate(outputs, start=1):
            safe_base = base.replace("/", "_").replace("\\", "_")
            save_name = f"{safe_base}_{i}.png"
            out.save(os.path.join(args.output_folder, save_name))
        print(f"[row {row_no}] Successfully generated {len(outputs)} images: {img_name}")

if __name__ == "__main__":
    main()