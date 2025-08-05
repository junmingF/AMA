#!/usr/bin/env python3
import argparse
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image

from pulid import attention_processor as attention
from pulid.pipeline_v1_1 import PuLIDPipeline
from pulid.utils import resize_numpy_image_long

torch.set_grad_enabled(False)

# Default model and sampler settings
BASE_DEFAULT = 'RunDiffusion/Juggernaut-XL-v9'
USE_LIGHTNING_DEFAULT = 'lightning' in BASE_DEFAULT.lower()
DEFAULT_CFG = 2.0 if USE_LIGHTNING_DEFAULT else 7.0
DEFAULT_STEPS = 5 if USE_LIGHTNING_DEFAULT else 25

DEFAULT_NEGATIVE_PROMPT = (
    'flaws in the eyes, flaws in the face, flaws, lowres, non-HDRi, low quality, worst quality,'
    'artifacts noise, text, watermark, glitch, deformed, mutated, ugly, disfigured, hands, '
    'low resolution, partially rendered objects, deformed or partially rendered eyes, '
    'deformed, deformed eyeballs, cross-eyed, blurry'
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch-generate images using PuLID from prompts and image names provided in an XLSX file."
    )
    parser.add_argument(
        '--base', type=str, default='RunDiffusion/Juggernaut-XL-v9',
        choices=[
            'Lykon/dreamshaper-xl-lightning',
            'RunDiffusion/Juggernaut-XL-v9',
        ],
        help="SDXL model repository"
    )
    parser.add_argument(
        '--image_folder', type=str, default="/disk1/fjm/img",
        help="Folder path containing reference images"
    )
    parser.add_argument(
        '--xlsx_file', type=str, default="/disk1/fjm/LLM/afterfliterprompt/i2p_sfw.xlsx",
        help="Path to the XLSX file (image_name in column 7, prompt in column 6)"
    )
    parser.add_argument(
        '--start_row', type=int, default=0,
        help="XLSX start row (1-based, inclusive)"
    )
    parser.add_argument(
        '--end_row', type=int, default=1700,
        help="XLSX end row (1-based, inclusive). Default: last row"
    )
    parser.add_argument(
        '--out_results_dir', type=str, default="/disk1/fjm/resultimg/pulid/i2p",
        help="Directory to save the generated images"
    )
    parser.add_argument(
        '--neg_prompt', type=str, default='',
        help="Global negative prompt"
    )
    parser.add_argument(
        '--scale', type=float, default=7,
        help="CFG (classifier-free guidance) scale"
    )
    parser.add_argument(
        '--seed', type=int, default=-1,
        help="Random seed (-1 for random each image)"
    )
    parser.add_argument(
        '--steps', type=int, default=25,
        help="Sampling steps"
    )
    parser.add_argument(
        '--H', type=int, default=1024,
        help="Image height"
    )
    parser.add_argument(
        '--W', type=int, default=1024,
        help="Image width"
    )
    parser.add_argument(
        '--id_scale', type=float, default=0.8,
        help="ID scale (higher improves ID fidelity but reduces editability)"
    )
    parser.add_argument(
        '--num_zero', type=int, default=20,
        help="num_zero (improves editability but reduces fidelity)"
    )
    parser.add_argument(
        '--ortho', choices=['off', 'v1', 'v2'], default='v2',
        help="attention ortho mode"
    )
    parser.add_argument(
        '--randomize_seed', action='store_true',
        help="Use a different random seed for each image"
    )
    parser.add_argument(
        "--cuda_device", type=int, default=1,
        help="Specify CUDA device index (e.g., 0 for cuda:0)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_results_dir, exist_ok=True)

    # Specify GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(args.cuda_device)
        device = f"cuda:{args.cuda_device}"
    else:
        device = "cpu"

    # Setup sampler and model
    args.sampler = 'dpmpp_sde' if 'lightning' in args.base.lower() else 'dpmpp_2m'
    pipeline = PuLIDPipeline(
        sdxl_repo="/disk1/fujm/Juggernaut-XL-v9",
        sampler=args.sampler,
        device=device   # Make sure PuLIDPipeline supports device parameter
    )

    # Read XLSX
    try:
        df = pd.read_excel(args.xlsx_file,  engine='openpyxl')
        print("Actual column names in the file:", df.columns.tolist())
    except Exception as e:
        print(f"Error reading XLSX: {e}", file=sys.stderr)
        sys.exit(1)
    total = len(df)
    start_idx = max(0, args.start_row - 1)
    end_idx = args.end_row if args.end_row and args.end_row <= total else total
    df = df.iloc[start_idx:end_idx]

    # Image generation loop
    for idx, row in df.iterrows():
        try:
            image_name = str(row['image_name']).strip()   # column 7
            prompt_text = str(row['prompt']).strip()      # column 6
        except Exception as e:
            print(f"[Row {idx+1}] Skipped: Failed to read image_name or prompt_text, {e}", file=sys.stderr)
            continue

        img_path = os.path.join(args.image_folder, image_name)
        if not os.path.isfile(img_path):
            print(f"[Row {idx+1}] Skipped: File not found {image_name}", file=sys.stderr)
            continue

        # Seed handling
        seed = random.randint(0, 2**32-1) if args.randomize_seed or args.seed < 0 else args.seed

        # Attention settings
        attention.NUM_ZERO = args.num_zero
        attention.ORTHO = (args.ortho == 'v1')
        attention.ORTHO_v2 = (args.ortho == 'v2')

        # Load ID image and compute embedding
        pil_img = Image.open(img_path).convert("RGB")
        np_img = np.array(pil_img)
        np_resized = resize_numpy_image_long(np_img, 1024)
        uncond_emb, id_emb = pipeline.get_id_embedding([np_resized])

        # Inference
        out_images = pipeline.inference(
            prompt_text,
            (1, args.H, args.W),
            args.neg_prompt,
            id_emb,
            uncond_emb,
            args.id_scale,
            args.scale,
            args.steps,
            int(seed)
        )
        # Save
        base, _ = os.path.splitext(image_name)
        safe_prompt = prompt_text.replace('/', '_')[:50]
        out_name = f"{idx-1:04d}_{base}_{safe_prompt}_seed{seed}.png"
        out_path = os.path.join(args.out_results_dir, out_name)
        # out_images[0] could be PIL.Image or numpy.array
        first = out_images[0]
        if isinstance(first, np.ndarray):
            Image.fromarray(first).save(out_path)
        else:
            first.save(out_path)
        print(f"[Row {idx-1}] Generated: {out_path}")

if __name__ == "__main__":
    main()