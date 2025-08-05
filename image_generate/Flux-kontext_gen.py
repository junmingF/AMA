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
        help="XLSX 文件路径，默认不含表头，第一列: image_name，第二列: prompt"
    )
    parser.add_argument(
        "--model_path", type=str, default="/disk1/fujm/FLUX.1-Kontext-dev",
        help="FluxKontextPipeline 模型本地路径或 HuggingFace 仓库路径"
    )
    parser.add_argument(
        "--image_folder", type=str, default="/disk1/fjm/img",
        help="输入图像所在文件夹"
    )
    parser.add_argument(
        "--output_folder", type=str, default="/disk1/fjm/resultimg/kontext/i2p",
        help="生成图像保存目录"
    )
    parser.add_argument(
        "--start_row", type=int, default=0,
        help="开始行（1-based，包含），默认 1"
    )
    parser.add_argument(
        "--end_row", type=int, default=1700,
        help="结束行（1-based，包含），默认到最后"
    )
    parser.add_argument(
        "--cuda_device", type=int, default=2,
        help="CUDA 设备索引"
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=2.5,
        help="指导尺度 (classifier-free guidance scale)"
    )
    parser.add_argument(
        "--num_outputs", type=int, default=1,
        help="每个输入生成的图像数量"
    )
    parser.add_argument(
        "--seed", type=int, default=9,
        help="随机种子（可指定，或与 --randomize_seed 一起使用）"
    )
    parser.add_argument(
        "--randomize_seed", action="store_true",
        help="为每张图随机种子"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    # 选择设备
    if torch.cuda.is_available():
        torch.cuda.set_device(args.cuda_device)
        device = f"cuda:{args.cuda_device}"
    else:
        print("警告: 未检测到 CUDA，使用 CPU 运行。")
        device = "cpu"

    # 选择数值精度
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device == "cpu":
        torch_dtype = torch.float32

    # 加载模型
    print(f"加载模型: {args.model_path} 到 {device}, dtype={torch_dtype}")
    pipe = FluxKontextPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype
    ).to(device)

    # 读取 XLSX
    try:
        df = pd.read_excel(args.xlsx_file,  engine="openpyxl")
    except Exception as e:
        print(f"读取 {args.xlsx_file} 失败: {e}")
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
            print(f"[第 {row_no} 行] 跳过：缺少 image_name 或 prompt")
            continue

        # 构造随机种子
        seed = torch.seed() if args.randomize_seed else args.seed
        generator = torch.Generator(device=device).manual_seed(seed)

        # 准备输入图像
        img_path = os.path.join(args.image_folder, img_name)
        if not os.path.isfile(img_path):
            print(f"[第 {row_no} 行] 跳过：文件不存在 {img_path}")
            continue
        pil_img = load_image(img_path)

        # 推理
        try:
            outputs = pipe(
                image=pil_img,
                prompt=prompt,
                guidance_scale=args.guidance_scale,
                num_images_per_prompt=args.num_outputs,
                generator=generator
            ).images
        except Exception as e:
            print(f"[第 {row_no} 行] 推理失败: {e}")
            continue

        # 保存
        base, _ = os.path.splitext(img_name)
        for i, out in enumerate(outputs, start=1):
            safe_base = base.replace("/", "_").replace("\\", "_")
            save_name = f"{safe_base}_{i}.png"
            out.save(os.path.join(args.output_folder, save_name))
        print(f"[第 {row_no} 行] 生成 {len(outputs)} 张图：{img_name}")

if __name__ == "__main__":
    main()

