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

# 默认模型库和采样器设置
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
        description="基于 PuLID 批量生成图像，通过 XLSX 提供 prompt 和 image_name。"
    )
    parser.add_argument(
        '--base', type=str, default='RunDiffusion/Juggernaut-XL-v9',
        choices=[
            'Lykon/dreamshaper-xl-lightning',
            'RunDiffusion/Juggernaut-XL-v9',
        ],
        help="SDXL 模型仓库"
    )
    parser.add_argument(
        '--image_folder', type=str, default="/disk1/fjm/img",
        help="参考照片文件夹路径"
    )
    parser.add_argument(
        '--xlsx_file', type=str, default="/disk1/fjm/LLM/afterfliterprompt/i2p_sfw.xlsx",
        help="XLSX 文件路径（图片名在第7列，prompt在第6列）"
    )
    parser.add_argument(
        '--start_row', type=int, default=0,
        help="XLSX 起始行（1-based，包含此行）"
    )
    parser.add_argument(
        '--end_row', type=int, default=1700,
        help="XLSX 结束行（1-based，包含此行），默认到最后"
    )
    parser.add_argument(
        '--out_results_dir', type=str, default="/disk1/fjm/resultimg/pulid/i2p",
        help="生成图片保存目录"
    )
    parser.add_argument(
        '--neg_prompt', type=str, default='',
        help="统一负面 Prompt"
    )
    parser.add_argument(
        '--scale', type=float, default=7,
        help="CFG 强度"
    )
    parser.add_argument(
        '--seed', type=int, default=-1,
        help="随机种子（-1 随机）"
    )
    parser.add_argument(
        '--steps', type=int, default=25,
        help="采样步数"
    )
    parser.add_argument(
        '--H', type=int, default=1024,
        help="图像高度"
    )
    parser.add_argument(
        '--W', type=int, default=1024,
        help="图像宽度"
    )
    parser.add_argument(
        '--id_scale', type=float, default=0.8,
        help="ID scale（提升 ID 保真度但降低可编辑性）"
    )
    parser.add_argument(
        '--num_zero', type=int, default=20,
        help="num zero（提升可编辑性但降低保真度）"
    )
    parser.add_argument(
        '--ortho', choices=['off', 'v1', 'v2'], default='v2',
        help="attention ortho 模式"
    )
    parser.add_argument(
        '--randomize_seed', action='store_true',
        help="对每张图使用随机种子"
    )
    parser.add_argument(
        "--cuda_device", type=int, default=1,
        help="指定 CUDA 设备编号（如 0 表示cuda:0）"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_results_dir, exist_ok=True)

    # 指定 GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(args.cuda_device)
        device = f"cuda:{args.cuda_device}"
    else:
        device = "cpu"

    # 设定采样器与模型
    args.sampler = 'dpmpp_sde' if 'lightning' in args.base.lower() else 'dpmpp_2m'
    pipeline = PuLIDPipeline(
        sdxl_repo="/disk1/fujm/Juggernaut-XL-v9",
        sampler=args.sampler,
        device=device   # 强烈建议确保PuLIDPipeline支持device参数
    )

    # 读取 XLSX
    try:
        df = pd.read_excel(args.xlsx_file,  engine='openpyxl')
        print("文件中的实际列名是:", df.columns.tolist()) # <--- 加上这一行来调试
    except Exception as e:
        print(f"Error reading XLSX: {e}", file=sys.stderr)
        sys.exit(1)
    total = len(df)
    start_idx = max(0, args.start_row - 1)
    end_idx = args.end_row if args.end_row and args.end_row <= total else total
    df = df.iloc[start_idx:end_idx]

    # 逐行生成
    for idx, row in df.iterrows():
        try:
            image_name = str(row['image_name']).strip()   # 第7列
            prompt_text = str(row['prompt']).strip()  # 第6列
        except Exception as e:
            print(f"[Row {idx+1}] 跳过：读取 image_name 或 prompt_text 失败，{e}", file=sys.stderr)
            continue

        img_path = os.path.join(args.image_folder, image_name)
        if not os.path.isfile(img_path):
            print(f"[Row {idx+1}] 跳过：未找到文件 {image_name}", file=sys.stderr)
            continue

        # 种子处理
        seed = random.randint(0, 2**32-1) if args.randomize_seed or args.seed < 0 else args.seed

        # attention 设置
        attention.NUM_ZERO = args.num_zero
        attention.ORTHO = (args.ortho == 'v1')
        attention.ORTHO_v2 = (args.ortho == 'v2')

        # 加载 ID 图并计算 embedding
        pil_img = Image.open(img_path).convert("RGB")
        np_img = np.array(pil_img)
        np_resized = resize_numpy_image_long(np_img, 1024)
        uncond_emb, id_emb = pipeline.get_id_embedding([np_resized])

        # 推理
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
        # 保存
        base, _ = os.path.splitext(image_name)
        safe_prompt = prompt_text.replace('/', '_')[:50]
        out_name = f"{idx-1:04d}_{base}_{safe_prompt}_seed{seed}.png"
        out_path = os.path.join(args.out_results_dir, out_name)
        # out_images[0] 可能为 PIL.Image 或 numpy.array
        first = out_images[0]
        if isinstance(first, np.ndarray):
            Image.fromarray(first).save(out_path)
        else:
            first.save(out_path)
        print(f"[Row {idx-1}] 已生成：{out_path}")

if __name__ == "__main__":
    main()