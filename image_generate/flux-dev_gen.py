import argparse
import os
import sys
import torch
import pandas as pd
from diffusers import FluxPipeline

def parse_args():
    """解析命令行参数，专为 FLUX.1-dev 模型定制。"""
    parser = argparse.ArgumentParser(
        description="使用 FLUX.1-dev 模型，根据XLSX文件中的提示词批量生成图像。"
    )
    # --- 文件与路径参数 ---
    parser.add_argument(
        "--xlsx_file", type=str, default='/disk1/fjm/LLM/0713_with_sentence-thking.xlsx',
        help="包含生成任务的XLSX文件路径 (必须包含表头)。"
    )
    parser.add_argument(
        "--model_path", type=str, default="/disk1/fujm/FLUX.1-dev",
        help="FLUX.1-dev 模型的本地路径或HuggingFace仓库名。"
    )
    parser.add_argument(
        "--output_folder", type=str, default="/disk1/fjm/base_ds/flux_dev/mis",
        help="用于保存生成图像的目录。"
    )
    # --- XLSX文件列定义 (MODIFIED: from index to name) ---
    parser.add_argument(
        "--prompt_col_name", type=str, default="sentence",
        help="提示词（Prompt）在XLSX文件中的'列名'。"
    )
    parser.add_argument(
        "--output_name_col", type=str, default="image_name",
        help="用作输出文件名的'列名'。"
    )
    # --- 批量处理参数 ---
    parser.add_argument(
        "--start_row", type=int, default=1,
        help="开始处理的数据行号（从1开始，不含表头）。"
    )
    parser.add_argument(
        "--end_row", type=int, default=None,
        help="结束处理的数据行号（从1开始，不含表头），默认为最后一行。"
    )
    # --- 模型与性能参数 ---
    parser.add_argument(
        "--cuda_device", type=int, default=2,
        help="要使用的CUDA设备索引号。"
    )
    parser.add_argument(
        "--cpu_offload", action="store_true",
        help="启用模型CPU卸载以节省显存。如果显存充足，无需开启此项。"
    )
    # --- FLUX 模型专用参数 ---
    parser.add_argument(
        "--width", type=int, default=512,
        help="生成图像的宽度。"
    )
    parser.add_argument(
        "--height", type=int, default=512,
        help="生成图像的高度。"
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=3.5,
        help="指导尺度（CFG Scale），FLUX模型建议值较低。"
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=50,
        help="推理步数。"
    )
    parser.add_argument(
        "--max_sequence_length", type=int, default=512,
        help="文本编码器的最大序列长度。"
    )
    parser.add_argument(
        "--num_outputs", type=int, default=1,
        help="每个提示词生成的图像数量。"
    )
    # --- 随机种子参数 ---
    parser.add_argument(
        "--seed", type=int, default=42,
        help="固定的随机种子，用于复现结果。"
    )
    parser.add_argument(
        "--randomize_seed", action="store_true",
        help="为每个任务使用不同的随机种子。"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    # 1. 设置设备和数据类型
    if torch.cuda.is_available():
        device = f"cuda:{args.cuda_device}"
        # FLUX 推荐使用 bfloat16
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        print("⚠️ 警告：未检测到 CUDA，将使用 CPU 运行。")
        device = "cpu"
        torch_dtype = torch.float32

    # 2. 加载模型
    print(f"正在加载 FLUX.1-dev 模型: {args.model_path}")
    try:
        pipe = FluxPipeline.from_pretrained(
            args.model_path,
            torch_dtype=torch_dtype
        )
        # 根据参数决定是否启用CPU卸载
        if args.cpu_offload and device != "cpu":
            print("...启用模型CPU卸载以节省显存。")
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)

    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        sys.exit(1)

    # 3. 读取 XLSX 文件 (MODIFIED: header=0 to use column names)
    try:
        df = pd.read_excel(args.xlsx_file, header=0, engine="openpyxl")
    except Exception as e:
        print(f"❌ 读取XLSX文件 {args.xlsx_file} 失败: {e}")
        sys.exit(1)

    # 4. 确定处理范围
    total_rows = len(df)
    start_index = max(0, args.start_row - 1)
    end_index = args.end_row if args.end_row and args.end_row <= total_rows else total_rows
    subset_df = df.iloc[start_index:end_index]
    print(f"\n准备处理从数据行 {start_index + 1} 到 {end_index} 的任务，共 {len(subset_df)} 条。")

    # 5. 循环处理每个任务
    for idx, row in subset_df.iterrows():
        # Excel中的实际行号 = pandas索引 + 2 (1 for header, 1 for 0-based index)
        current_row_num = idx + 2
        
        try:
            # MODIFIED: Get data by column name
            prompt = str(row[args.prompt_col_name]).strip()
            output_base_name = str(row[args.output_name_col]).strip()
        except KeyError as e:
            # MODIFIED: More informative error message
            print(f"⏭️ 跳过第 {current_row_num} 行：列名 {e} 不存在。请检查 --prompt_col_name 和 --output_name_col 参数。")
            print(f"    可用的列名有: {list(df.columns)}")
            continue

        if not prompt or pd.isna(prompt) or not output_base_name or pd.isna(output_base_name):
            print(f"⏭️ 跳过第 {current_row_num} 行：缺少提示词或输出文件名。")
            continue

        # 6. 设置随机种子
        seed = torch.seed() if args.randomize_seed else args.seed
        # 注意：FLUX模型在CPU卸载模式下可能需要将生成器也放在CPU上
        generator_device = "cpu" if args.cpu_offload and device != "cpu" else device
        generator = torch.Generator(device=generator_device).manual_seed(seed)

        # 7. 推理
        print(f"▶️ 正在生成第 {current_row_num} 行: '{prompt[:80]}...'")
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
            print(f"❌ 第 {current_row_num} 行推理失败: {e}")
            continue

        # 8. 保存图像
        base_name, _ = os.path.splitext(output_base_name)
        for i, image_out in enumerate(images, start=1):
            safe_base_name = base_name.replace("/", "_").replace("\\", "_")
            save_name = f"{safe_base_name}_{i}.png" if args.num_outputs > 1 else f"{safe_base_name}.png"
            save_path = os.path.join(args.output_folder, save_name)
            image_out.save(save_path)
        
        print(f"✅ 成功为第 {current_row_num} 行生成了 {len(images)} 张图像。")

if __name__ == "__main__":
    main()