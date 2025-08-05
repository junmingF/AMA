
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
        description="Batch-generate images with InfiniteYou-FLUX pipeline based on prompts in a CSV file."
    )
    parser.add_argument(
        "--image_folder", type=str, default="/disk1/fjm/img",
        help="包含所有参考照片的文件夹路径"
    )
    parser.add_argument(
        "--csv_file", type=str, default="/disk1/fjm/clipscore/afterfliterprompt/mma_sfw.xlsx",
        help="CSV 文件路径，第一列 image_name，对应照片文件名；第二列 prompt，对应生成提示"
    )

    parser.add_argument(
        "--start_row", type=int, default=0,
        help="处理 CSV 的起始行（1-based，包含此行），默认 1"
    )
    parser.add_argument(
        "--end_row", type=int, default=1700,
        help="处理 CSV 的结束行（1-based，包含此行），默认处理到最后一行"
    )
    parser.add_argument(
        "--out_results_dir", type=str, default="/disk1/fjm/resultimg/infinite/mma",
        help="生成图片保存目录"
    )
    parser.add_argument("--base_model_path", type=str,
        default="/disk1/fujm/FLUX.1-dev",
        help="FLUX 基础模型路径"
    )
    parser.add_argument(
        "--infu_flux_version", default="v1.0",
        help="InfiniteYou-FLUX 版本 (目前仅 v1.0)"
    )
    parser.add_argument(
        "--model_version", default="aes_stage2",
        choices=["aes_stage2", "sim_stage1"],
        help="模型阶段版本"
    )
    parser.add_argument("--cuda_device", default=0, type=int,
        help="使用的 CUDA 设备编号"
    )
    parser.add_argument("--seed", default=9, type=int,
        help="随机种子 (0 表示每张图随机)")
    parser.add_argument("--guidance_scale", default=3.5, type=float,
        help="classifier-free guidance 强度"
    )
    parser.add_argument("--num_steps", default=30, type=int,
        help="扩散步数"
    )
    parser.add_argument("--infusenet_conditioning_scale", default=1.0, type=float)
    parser.add_argument("--infusenet_guidance_start", default=0.0, type=float)
    parser.add_argument("--infusenet_guidance_end", default=1.0, type=float)
    parser.add_argument("--enable_realism_lora", action="store_true",
        help="启用 realism LoRA"
    )
    parser.add_argument("--enable_anti_blur_lora", action="store_true",
        help="启用 anti_blur LoRA"
    )
    parser.add_argument("--quantize_8bit", action="store_true",
        help="启用 8-bit 量化"
    )
    parser.add_argument("--cpu_offload", action="store_true",
        help="启用 CPU offload 减少显存占用"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_results_dir, exist_ok=True)

    # Set CUDA device
    torch.cuda.set_device(args.cuda_device)
    device = f"cuda:{args.cuda_device}"

    # 加载 InfiniteYou-FLUX 管道
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

    # 可选 LoRA
    lora_dir = "/disk1/fujm/InfiniteYou_model/supports/optional_loras"
    if not os.path.isdir(lora_dir):
        lora_dir = "./models/InfiniteYou/supports/optional_loras"
    loras = []
    if args.enable_realism_lora:
        loras.append([os.path.join(lora_dir, "flux_realism_lora.safetensors"), "realism", 1.0])
    if args.enable_anti_blur_lora:
        loras.append([os.path.join(lora_dir, "flux_anti_blur_lora.safetensors"), "anti_blur", 1.0])
    pipe.load_loras(loras)
  
    # 读取 XLSX（不用读 csv 了）
    try:
        df = pd.read_excel(args.csv_file, dtype=str, engine="openpyxl")
    except Exception as e:
        print(f"读取 xlsx 文件出错: {e}")
        sys.exit(1)

    total = len(df)
    start_idx = max(args.start_row - 1, 0)
    end_idx = args.end_row if args.end_row and args.end_row <= total else total
    df = df.iloc[start_idx:end_idx]

    for i, row in df.iterrows():
        try:
            image_name = str(row['image_name']).strip()  # Excel第7列
            prompt_text = str(row['prompt']).strip() # Excel第6列
        except Exception:
            print(f"[Row {i+1}] 跳过：未能正确读取 image_name 或 prompt_text")
            continue

        img_path = os.path.join(args.image_folder, image_name)
        if not os.path.isfile(img_path):
            print(f"[Row {i+1}] 跳过：未找到文件 {image_name}")
            continue

        # 准备随机种子
        seed = random.randint(0, 0xFFFFFFFF) if args.seed == 0 else args.seed
        gen = torch.Generator(device=device).manual_seed(seed)

        # 打开图像并调用管道
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

        # 保存结果
        base, _ = os.path.splitext(image_name)
        safe_prompt = prompt_text.replace("/", "_")[:50]
        out_name = f"{base}_{safe_prompt}_seed{seed}.png"
        out_path = os.path.join(args.out_results_dir, out_name)
        out.save(out_path)
        print(f"[Row {i+1}] 生成：{out_path}")

if __name__ == "__main__":
    main()
