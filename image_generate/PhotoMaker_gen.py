#!/usr/bin/env python3
import argparse
import os
import sys
import random

import numpy as np
import pandas as pd
import torch
from diffusers.utils import load_image
from diffusers import EulerDiscreteScheduler, T2IAdapter

from photomaker import PhotoMakerStableDiffusionXLAdapterPipeline
from photomaker import FaceAnalysis2, analyze_faces
from style_template import styles
from aspect_ratio_template import aspect_ratios

MAX_SEED = np.iinfo(np.int32).max

def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch-generate images using PhotoMaker pipeline based on prompts in an XLSX file."
    )
    parser.add_argument("--csv_file", type=str, default="/disk1/fjm/UniPortrait/assets/faceprompt.xlsx",
                        help="Path to XLSX file with at least 7 columns (image_name in column 7, prompt in column 6)")
    parser.add_argument("--image_folder", type=str, default="/disk1/fjm/img",
                        help="Folder containing input images referenced in XLSX")
    parser.add_argument("--output_folder", type=str, default="/disk1/fjm/resultimg",
                        help="Directory to save generated images")
    parser.add_argument("--start_row", type=int, default=0,
                        help="1-based index of the first row to process (inclusive)")
    parser.add_argument("--end_row", type=int, default=5,
                        help="1-based index of the last row to process (inclusive). Default=last row")
    parser.add_argument("--style_name", choices=list(styles.keys()),
                        default="Photographic (Default)", help="Style template name")
    parser.add_argument("--aspect_ratio", choices=list(aspect_ratios.keys()),
                        default=list(aspect_ratios.keys())[0], help="Output aspect ratio label")
    parser.add_argument("--negative_prompt", type=str, default=(
        "lowres, bad anatomy, bad hands, text, error, missing fingers, "
        "extra digit, fewer digits, cropped, worst quality, low quality, normal quality, "
        "jpeg artifacts, signature, watermark, username, blurry"
    ), help="Negative prompt to apply to all generations")
    parser.add_argument("--num_steps", type=int, default=50,
                        help="Number of diffusion sampling steps")
    parser.add_argument("--style_strength_ratio", type=int, default=20,
                        help="Style strength ratio (%)")
    parser.add_argument("--num_outputs", type=int, default=2,
                        help="Number of images to generate per input")
    parser.add_argument("--guidance_scale", type=float, default=5.0,
                        help="Classifier-free guidance scale")
    parser.add_argument("--seed", type=int, default=9,
                        help="Random seed (overridden if --randomize_seed is set)")
    parser.add_argument("--randomize_seed", action="store_true",
                        help="If set, pick a random seed for each image")
    parser.add_argument("--base_model_path", type=str,
                        default="/disk1/fujm/photomaker_model/RealVisXL_V4.0",
                        help="Path to the base SDXL model")
    parser.add_argument("--adapter_path", type=str,
                        default="/disk1/fujm/photomaker_model/t2i-adapter-sketch-sdxl-1.0",
                        help="Path to the T2I-Adapter checkpoint directory")
    parser.add_argument("--photomaker_ckpt", type=str,
                        default="/disk1/fujm/photomaker_model/PhotoMaker-V2/photomaker-v2.bin",
                        help="Path to the PhotoMaker adapter .bin file")
    return parser.parse_args()

def apply_style(style_name: str, positive: str, negative: str):
    base_p, base_n = styles.get(style_name, styles["Photographic (Default)"])
    prompt = base_p.replace("{prompt}", positive)
    neg = base_n + " " + negative
    return prompt, neg

def main():
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    # Device and dtype
    if torch.cuda.is_available():
        device = "cuda"
    elif sys.platform == "darwin" and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device == "mps":
        torch_dtype = torch.float16

    # Prepare face detector
    face_detector = FaceAnalysis2(
        providers=['CUDAExecutionProvider'],
        allowed_modules=['detection', 'recognition']
    )
    face_detector.prepare(ctx_id=0, det_size=(640, 640))

    # Load adapter and pipeline
    adapter = T2IAdapter.from_pretrained(
        args.adapter_path, torch_dtype=torch_dtype, variant="fp16"
    ).to(device)
    pipe = PhotoMakerStableDiffusionXLAdapterPipeline.from_pretrained(
        args.base_model_path,
        adapter=adapter,
        torch_dtype=torch_dtype,
        use_safetensors=True,
        variant="fp16"
    ).to(device)
    pipe.load_photomaker_adapter(
        os.path.dirname(args.photomaker_ckpt),
        subfolder="",
        weight_name=os.path.basename(args.photomaker_ckpt),
        trigger_word="img",
        pm_version="v2",
    )
    pipe.id_encoder.to(device)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.fuse_lora()
    pipe.to(device)

    # 读取XLSX（无表头推测。每一行下标从0开始，第6列=row[5]为prompt，第7列=row[6]为图片名）
    try:
        df = pd.read_excel(args.csv_file, header=None, engine="openpyxl")
    except Exception as e:
        print(f"读取 xlsx 文件出错: {e}")
        sys.exit(1)
    total_rows = len(df)
    start = max(0, args.start_row - 1)
    end = args.end_row if args.end_row and args.end_row <= total_rows else total_rows
    df = df.iloc[start:end]

    for idx, row in df.iterrows():
        try:
            image_name = str(row[6])      # 第7列
            prompt_text = str(row[5])     # 第6列
        except Exception:
            print(f"[Row {idx+1}] 跳过：读取 image_name 或 prompt_text 失败")
            continue

        # Setting seed
        seed = random.randint(0, MAX_SEED) if args.randomize_seed else args.seed
        generator = torch.Generator(device=device).manual_seed(seed)

        # 检查trigger word
        image_token_id = pipe.tokenizer.convert_tokens_to_ids(pipe.trigger_word)
        input_ids = pipe.tokenizer.encode(prompt_text)
        if image_token_id not in input_ids:
            print(f"[Row {idx+1}] Skipping '{image_name}': missing trigger word 'img'")
            continue
        if input_ids.count(image_token_id) > 1:
            print(f"[Row {idx+1}] Skipping '{image_name}': multiple trigger words")
            continue

        img_path = os.path.join(args.image_folder, image_name)
        if not os.path.isfile(img_path):
            print(f"[Row {idx+1}] Skipping '{image_name}': file not found")
            continue
        pil_img = load_image(img_path)
        np_img = np.array(pil_img)[:, :, ::-1]
        faces = analyze_faces(face_detector, np_img)
        if not faces:
            print(f"[Row {idx+1}] Skipping '{image_name}': no face detected")
            continue
        id_embed = torch.from_numpy(faces[0]['embedding']).unsqueeze(0).to(device)

        # 准备prompt、aspect ratio、start_merge_step
        styled_p, styled_n = apply_style(args.style_name, prompt_text, args.negative_prompt)
        width, height = aspect_ratios[args.aspect_ratio]
        start_merge = int(args.style_strength_ratio / 100 * args.num_steps)
        start_merge = min(start_merge, 30)

        outputs = pipe(
            prompt=styled_p,
            negative_prompt=styled_n,
            input_id_images=[pil_img],
            id_embeds=id_embed,
            width=width,
            height=height,
            num_inference_steps=args.num_steps,
            num_images_per_prompt=args.num_outputs,
            guidance_scale=args.guidance_scale,
            start_merge_step=start_merge,
            generator=generator,
        ).images

        base, _ = os.path.splitext(image_name)
        for i, out in enumerate(outputs, start=1):
            # 文件名安全处理，防止太长或包含特殊符号
            safe_prompt = styled_p.replace("/", "_").replace("\\", "_")[:40]
            save_path = os.path.join(args.output_folder, f"{i}_{base}_{safe_prompt}.png")
            out.save(save_path)
        print(f"[Row {idx+1}] Generated {len(outputs)} images for '{image_name}'")

if __name__ == "__main__":
    main()