import argparse
import os
import sys
import torch
import pandas as pd
from diffusers import StableDiffusionPipeline

def parse_args():
    """
    解析命令行参数。
    """
    parser = argparse.ArgumentParser(
        description="使用StableDiffusionPipeline根据XLSX文件中的prompts批量生成适合打印的高质量图像。"
    )
    # --- 文件与路径参数 ---
    parser.add_argument(
        "--xlsx_file", type=str, default='/disk1/fjm/LLM/0713_with_sentence-thking.xlsx', # MODIFIED: Made required as it's essential
        help="包含生成任务的XLSX文件路径 (必须包含表头)"
    )
    parser.add_argument(
        "--model_path", type=str, default="/disk1/fujm/stable-diffusion-v1-5",
        help="Stable Diffusion v1.5 模型本地路径或HuggingFace仓库名"
    )
    parser.add_argument(
        "--output_folder", type=str, default="/disk1/fjm/base_ds/sd1-5/mis", # MODIFIED: More generic default
        help="生成图像的保存目录"
    )
    # --- XLSX文件内容定义 (MODIFIED: from index to name) ---
    parser.add_argument(
        "--prompt_col_name", type=str, default="sentence",
        help="Prompt在XLSX文件中的'列名'"
    )
    parser.add_argument(
        "--output_name_col", type=str, default="image_name",
        help="用作输出文件名的'列名'"
    )
    # --- 批量处理参数 ---
    parser.add_argument(
        "--start_row", type=int, default=1,
        help="开始处理的数据行号（1-based，不含表头）"
    )
    parser.add_argument(
        "--end_row", type=int, default=None,
        help="结束处理的数据行号（1-based，不含表头），默认为最后一行"
    )
    # --- 模型与性能参数 ---
    parser.add_argument(
        "--cuda_device", type=int, default=0, help="要使用的CUDA设备索引"
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=7.5, help="指导尺度 (CFG scale)"
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=50, help="推理步数"
    )
    parser.add_argument(
        "--num_outputs", type=int, default=1, help="每个Prompt生成的图像数量"
    )
    # --- 分辨率参数 ---
    parser.add_argument(
        "--width", type=int, default=512, help="生成图像的宽度"
    )
    parser.add_argument(
        "--height", type=int, default=512, help="生成图像的高度"
    )
    # --- 随机种子参数 ---
    parser.add_argument(
        "--seed", type=int, default=9, help="固定的随机种子"
    )
    parser.add_argument(
        "--randomize_seed", action="store_true", help="为每个任务使用不同的随机种子"
    )
    # --- Prompt增强参数 ---
    parser.add_argument(
        "--quality_prompt", type=str, 
        default="masterpiece, best quality, high resolution, photorealistic, 8k, UHD, detailed, sharp focus, professional photography",
        help="附加到每个prompt末尾的质量增强词"
    )
    parser.add_argument(
        "--negative_prompt", type=str,
        default="(worst quality, low quality:1.4), (blurry:1.2), ugly, deformed, disfigured, bad anatomy, extra limbs, fused fingers, too many fingers, malformed hands, text, watermark, signature, username, logo, jpeg artifacts, lowres",
        help="通用的负面提示"
    )
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    # 1. 选择设备
    device = f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("警告: 未检测到 CUDA，将使用 CPU 运行。")

    # 2. 选择数值精度
    torch_dtype = torch.float16 if device != "cpu" else torch.float32

    # 3. 加载模型
    print(f"加载模型: {args.model_path} 到 {device}, dtype={torch_dtype}")
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            args.model_path,
            torch_dtype=torch_dtype
        ).to(device)
    except Exception as e:
        print(f"加载模型失败: {e}")
        sys.exit(1)

    # 4. 读取 XLSX 文件 (MODIFIED: header=0 to use first row as column names)
    try:
        df = pd.read_excel(args.xlsx_file, header=0, engine="openpyxl")
    except Exception as e:
        print(f"读取XLSX文件 {args.xlsx_file} 失败: {e}")
        sys.exit(1)

    # 5. 确定处理范围 (Note: user provides 1-based row numbers for data rows)
    total_rows = len(df)
    start_index = max(0, args.start_row - 1)
    end_index = args.end_row if args.end_row and args.end_row <= total_rows else total_rows
    subset_df = df.iloc[start_index:end_index]

    print(f"准备处理从数据行 {start_index + 1} 到 {end_index} 的任务，共 {len(subset_df)} 条。")
    print(f"打印质量设置: 分辨率={args.width}x{args.height}, CFG={args.guidance_scale}, Steps={args.num_inference_steps}")

    # 6. 循环处理每一行任务
    for idx, row in subset_df.iterrows():
        # current_row_num is the actual excel row number (1-based header + 1-based index)
        current_row_num = idx + 2 
        
        try:
            # MODIFIED: Get data by column name
            prompt_base = str(row[args.prompt_col_name]).strip()
            output_base_name = str(row[args.output_name_col]).strip()
        except KeyError as e:
            # MODIFIED: More informative error message
            print(f"[第 {current_row_num} 行] 跳过：列名 {e} 不存在。请检查 --prompt_col_name 和 --output_name_col 参数。")
            print(f"    可用的列名有: {list(df.columns)}")
            continue

        if not prompt_base or pd.isna(prompt_base) or not output_base_name or pd.isna(output_base_name):
            print(f"[第 {current_row_num} 行] 跳过：缺少 Prompt 或输出文件名。")
            continue
            
        full_prompt = f"{prompt_base}, {args.quality_prompt}"

        # 7. 构造随机种子
        seed = torch.seed() if args.randomize_seed else args.seed
        generator = torch.Generator(device=device).manual_seed(seed)

        # 8. 推理
        print(f"[第 {current_row_num} 行] 正在生成... Prompt: '{prompt_base[:80]}...'")
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
            print(f"[第 {current_row_num} 行] 推理失败: {e}")
            continue

        # 9. 保存图像
        base_name, _ = os.path.splitext(output_base_name)
        for i, image_out in enumerate(outputs, start=1):
            safe_base_name = base_name.replace("/", "_").replace("\\", "_")
            save_name = f"{safe_base_name}_{i}.png" if args.num_outputs > 1 else f"{safe_base_name}.png"
            save_path = os.path.join(args.output_folder, save_name)
            image_out.save(save_path)
        
        print(f"[第 {current_row_num} 行] 成功生成 {len(outputs)} 张图像，保存在 {args.output_folder}")

if __name__ == "__main__":
    main()