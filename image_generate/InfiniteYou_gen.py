#!/usr/bin/env python3
import argparse
import os
import sys
import random

import pandas as pd
import torch
from PIL import Image

from pipelines.pipeline_infu_flux import InfUFluxPipeline

def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch generate images with InfiniteYou-FLUX pipeline based on prompts in an Excel file."
    )
    parser.add_argument(
        "--image_folder", type=str, default="/disk1/fjm/img",
        help="Path to the folder containing all reference photos."
    )
    parser.add_argument(
        "--csv_file", type=str, default="/disk1/fjm/clipscore/afterfliterprompt/mma_sfw.xlsx",
        help="Path to the Excel file. First column: image_name (photo file name); Second column: prompt (text prompt for generation)."
    )
    parser.add_argument(
        "--start_row", type=int, default=0,
        help="Index of the starting row in the Excel file to process (1-based, inclusive). Default: 1"
    )
    parser.add_argument(
        "--end_row", type=int, default=1700,
        help="Index of the ending row in the Excel file to process (1-based, inclusive). Default: last row"
    )
    parser.add_argument(
        "--out_results_dir", type=str, default="/disk1/fjm/resultimg/infinite/mma",
        help="Directory to save the generated images."
    )
    parser.add_argument(
        "--base_model_path", type=str, default="/disk1/fujm/FLUX.1-dev",
        help="Path to the base FLUX model."
    )
    parser.add_argument(
        "--infu_flux_version", default="v1.0",
        help="InfiniteYou-FLUX version (currently only v1.0)."
    )
    parser.add_argument(
        "--model_version", default="aes_stage2",
        choices=["aes_stage2", "sim_stage1"],
        help="Version of model stage."
    )
    parser.add_argument(
        "--cuda_device", default=0, type=int,
        help="CUDA device index to use."
    )
    parser.add_argument(
        "--seed", default=9, type=int,
        help="Random seed (0 means random per image)."
    )
    parser.add_argument(
        "--guidance_scale", default=3.5, type=float,
        help="classifier-free guidance strength."
    )
    parser.add_argument(
        "--num_steps", default=30, type=int,
        help="Number of diffusion steps."
    )
    parser.add_argument(
        "--infusenet_conditioning_scale", default=1.0, type=float,
        help="InfuseNet conditioning scale."
    )
    parser.add_argument(
        "--infusenet_guidance_start", default=0.0, type=float,
        help="InfuseNet guidance start."
    )
    parser.add_argument(
        "--infusenet_guidance_end", default=1.0, type=float,
        help="InfuseNet guidance end."
    )
    parser.add_argument(
        "--enable_realism_lora", action="store_true",
        help="Enable realism LoRA."
    )
    parser.add_argument(
        "--enable_anti_blur_lora", action="store_true",
        help="Enable anti_blur LoRA."
    )
    parser.add_argument(
        "--quantize_8bit", action="store_true",
        help="Enable 8-bit quantization."
    )
    parser.add_argument(
        "--cpu_offload", action="store_true",
        help="Enable CPU offload to reduce VRAM usage."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_results_dir, exist_ok=True)

    # Set CUDA device
    torch.cuda.set_device(args.cuda_device)
    device = f"cuda:{args.cuda_device}"

    # Load InfiniteYou-FLUX pipeline
    infu_model_path = os.path.join(
        "/disk1/fujm/InfiniteYou_model",
        f"infu_flux_{args.infu_flux_version}",
        args.model_version
    )
    insightface_root = "/disk1/fujm/InfiniteYou_model/supports/insightface"

    pipe = InfUFluxPipeline(
        base_model_path=args.base_model_path,
        infu_model_path=infu_model_path,
        insightface_root_path=insightface_root,
        infu_flux_version=args.infu_flux_version,
        model_version=args.model_version,
        quantize_8bit=args.quantize_8bit,
        cpu_offload=args.cpu_offload,
    )

    # Optional LoRA
    lora_dir = "/disk1/fujm/InfiniteYou_model/supports/optional_loras"
    if not os.path.isdir(lora_dir):
        lora_dir = "./models/InfiniteYou/supports/optional_loras"
    loras = []
    if args.enable_realism_lora:
        loras.append([os.path.join(lora_dir, "flux_realism_lora.safetensors"), "realism", 1.0])
    if args.enable_anti_blur_lora:
        loras.append([os.path.join(lora_dir, "flux_anti_blur_lora.safetensors"), "anti_blur", 1.0])
    pipe.load_loras(loras)
  
    # Read Excel file (not CSV)
    try:
        df = pd.read_excel(args.csv_file, dtype=str, engine="openpyxl")
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        sys.exit(1)

    total = len(df)
    start_idx = max(args.start_row - 1, 0)
    end_idx = args.end_row if args.end_row and args.end_row <= total else total
    df = df.iloc[start_idx:end_idx]

    for i, row in df.iterrows():
        try:
            image_name = str(row['image_name']).strip()
            prompt_text = str(row['prompt']).strip()
        except Exception:
            print(f"[Row {i+1}] Skipped: Unable to correctly read image_name or prompt_text")
            continue

        img_path = os.path.join(args.image_folder, image_name)
        if not os.path.isfile(img_path):
            print(f"[Row {i+1}] Skipped: file not found {image_name}")
            continue

        # Handle random seed
        seed = random.randint(0, 0xFFFFFFFF) if args.seed == 0 else args.seed
        gen = torch.Generator(device=device).manual_seed(seed)

        # Open image and call pipeline
        img = Image.open(img_path).convert("RGB")
        out = pipe(
            id_image=img,
            prompt=prompt_text,
            seed=seed,
            guidance_scale=args.guidance_scale,
            num_steps=args.num_steps,
            infusenet_conditioning_scale=args.infusenet_conditioning_scale,
            infusenet_guidance_start=args.infusenet_guidance_start,
            infusenet_guidance_end=args.infusenet_guidance_end,
            cpu_offload=args.cpu_offload,
        )

        # Save result
        base, _ = os.path.splitext(image_name)
        safe_prompt = prompt_text.replace("/", "_")[:50]
        out_name = f"{base}_{safe_prompt}_seed{seed}.png"
        out_path = os.path.join(args.out_results_dir, out_name)
        out.save(out_path)
        print(f"[Row {i+1}] Generated: {out_path}")

if __name__ == "__main__":
    main()