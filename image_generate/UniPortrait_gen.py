import os  
import pandas as pd 
import argparse  
from io import BytesIO  
import cv2  
import numpy as np  
import torch  
from PIL import Image  
from tqdm import tqdm  

# 导入之前定义的函数  
from uniportrait import inversion  
from uniportrait.uniportrait_attention_processor import attn_args  
from uniportrait.uniportrait_pipeline import UniPortraitPipeline  
from insightface.app import FaceAnalysis  
from insightface.utils import face_align  
from diffusers import DDIMScheduler, AutoencoderKL, ControlNetModel, StableDiffusionControlNetPipeline  

# 定义填充图像的函数  
def pad_np_bgr_image(np_image, scale=1.25):  
    assert scale >= 1.0, "scale should be >= 1.0"  
    pad_scale = scale - 1.0  
    h, w = np_image.shape[:2]  
    top = bottom = int(h * pad_scale)  
    left = right = int(w * pad_scale)  
    ret = cv2.copyMakeBorder(np_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(128, 128, 128))  
    return ret, (left, top)  # 返回填充后的图像 ret 以及左上角的偏移坐标 (left, top)  

# 定义处理面部图像的函数  
def process_faceid_image(pil_faceid_image, face_app):  
    np_faceid_image = np.array(pil_faceid_image.convert("RGB"))  
    img = cv2.cvtColor(np_faceid_image, cv2.COLOR_RGB2BGR)  
    faces = face_app.get(img)  # bgr  
    if len(faces) == 0:  
        # padding, try again  
        _h, _w = img.shape[:2]  
        _img, left_top_coord = pad_np_bgr_image(img)  
        faces = face_app.get(_img)  
        if len(faces) == 0:  
            print("Warning: No face detected in the image. Continue processing...")  
            return None  

        min_coord = np.array([0, 0])  
        max_coord = np.array([_w, _h])  
        sub_coord = np.array([left_top_coord[0], left_top_coord[1]])  
        for face in faces:  
            face.bbox = np.minimum(np.maximum(face.bbox.reshape(-1, 2) - sub_coord, min_coord), max_coord).reshape(4)  
            face.kps = face.kps - sub_coord  

    faces = sorted(faces, key=lambda x: abs((x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1])), reverse=True)  
    if len(faces) == 0:  
        print("Warning: No face detected after sorting.")  
        return None  
    faceid_face = faces[0]  
    norm_face = face_align.norm_crop(img, landmark=faceid_face.kps, image_size=224)  
    pil_faceid_align_image = Image.fromarray(cv2.cvtColor(norm_face, cv2.COLOR_BGR2RGB))  

    return pil_faceid_align_image  

# 定义准备条件的函数  
def prepare_single_faceid_cond_kwargs(pil_faceid_image=None, pil_faceid_supp_images=None,  
                                      pil_faceid_mix_images=None, mix_scales=None, face_app=None):  
    pil_faceid_align_images = []  
    # 处理主要的面部图像并将对齐后的图像加入 pil_faceid_align_images 列表。  
    if pil_faceid_image:  
        processed_image = process_faceid_image(pil_faceid_image, face_app)  
        if processed_image:  
            pil_faceid_align_images.append(processed_image)  
    # 遍历辅助图像列表并处理每个图像，将对齐结果追加到 pil_faceid_align_images。  
    if pil_faceid_supp_images and len(pil_faceid_supp_images) > 0:  
        for pil_faceid_supp_image in pil_faceid_supp_images:  
            if isinstance(pil_faceid_supp_image, Image.Image):  
                processed_image = process_faceid_image(pil_faceid_supp_image, face_app)  
            else:  
                processed_image = process_faceid_image(Image.open(BytesIO(pil_faceid_supp_image)), face_app)  
            if processed_image:  
                pil_faceid_align_images.append(processed_image)  

    mix_refs = []  
    mix_ref_scales = []  
    # 遍历混合图像列表，分别处理每个图像并存储到 mix_refs，同时记录对应的混合比例到 mix_ref_scales。  
    if pil_faceid_mix_images and mix_scales:  
        for pil_faceid_mix_image, mix_scale in zip(pil_faceid_mix_images, mix_scales):  
            if pil_faceid_mix_image:  
                processed_mix_image = process_faceid_image(pil_faceid_mix_image, face_app)  
                if processed_mix_image:  
                    mix_refs.append(processed_mix_image)  
                    mix_ref_scales.append(mix_scale)  

    single_faceid_cond_kwargs = None  
    if len(pil_faceid_align_images) > 0:  
        single_faceid_cond_kwargs = {  
            "refs": pil_faceid_align_images  
        }  
        if len(mix_refs) > 0:  
            single_faceid_cond_kwargs["mix_refs"] = mix_refs  
            single_faceid_cond_kwargs["mix_scales"] = mix_ref_scales  

    return single_faceid_cond_kwargs  

# 定义生成图像的函数  
def text_to_single_id_generation_process(  
        uniportrait_pipeline,  # 将 uniportrait_pipeline 作为参数传入  
        pil_faceid_image=None, pil_faceid_supp_images=None, 
        pil_ip_image=None, 
        pil_faceid_mix_image_1=None, mix_scale_1=0.0,  
        pil_faceid_mix_image_2=None, mix_scale_2=0.0,  
        faceid_scale=1.0, face_structure_scale=0.5,  
        prompt="", negative_prompt="nsfw",  
        num_samples=1, seed=-1,  
        image_resolution="512x512",  
        inference_steps=30,  
        face_app=None  
 ):  
    if seed == -1:  
        seed = None  

    single_faceid_cond_kwargs = prepare_single_faceid_cond_kwargs(  
        pil_faceid_image,  
        pil_faceid_supp_images,  
        [pil_faceid_mix_image_1, pil_faceid_mix_image_2],  
        [mix_scale_1, mix_scale_2],  
        face_app  
    )  

    cond_faceids = [single_faceid_cond_kwargs] if single_faceid_cond_kwargs else []  

    # 重置注意力参数  
    attn_args.reset()  
    # 设置面部条件  
    attn_args.lora_scale = 1.0 if len(cond_faceids) == 1 else 0.0  # single-faceid lora  
    attn_args.multi_id_lora_scale = 1.0 if len(cond_faceids) > 1 else 0.0  # multi-faceid lora  
    attn_args.faceid_scale = faceid_scale if len(cond_faceids) > 0 else 0.0  
    attn_args.num_faceids = len(cond_faceids)  
    print(attn_args)  

    h, w = map(int, image_resolution.split("x"))  
    prompt = [prompt] * num_samples  
    negative_prompt = [negative_prompt] * num_samples  
    images = uniportrait_pipeline.generate(  
        prompt=prompt,  
        negative_prompt=negative_prompt, 
        pil_ip_image=pil_ip_image, 
        cond_faceids=cond_faceids,  
        face_structure_scale=face_structure_scale,  
        seed=seed,  
        guidance_scale=7.5,  
        num_inference_steps=inference_steps,  
        image=[torch.zeros([1, 3, h, w])],  
        controlnet_conditioning_scale=[0.0]  
    )  
    final_out = []  
    for pil_image in images:  
        final_out.append(pil_image)  

    return final_out  

 
   
if __name__ == "__main__":  

    parser = argparse.ArgumentParser(description="使用 UniPortrait 生成图像的脚本。")  
    parser.add_argument("--device", type=str, default="2", help="指定要使用的 GPU 设备，例如 '0' 或 '0,1'。")  
    parser.add_argument("--start_row", type=int, default=0, help="CSV 文件中要处理的起始行索引（从 0 开始）。")  
    parser.add_argument("--end_row", type=int, default=10, help="CSV 文件中要处理的结束行索引（包含）。")  
    parser.add_argument("--faceid_scale", type=float, default=0.8, help="决定人脸多像参考的图像。")
    parser.add_argument("--face_structure_scale", type=float, default=0.4, help="决定姿势多像参考的图像。")
    parser.add_argument("--image_dir", type=str, default="/disk1/fjm/img", help="包含输入图像的目录。") 
    parser.add_argument("--prompt_xlsx", type=str, default="/disk1/fjm/LLM/new_prompt/output/full_0803_0-2000.xlsx", help="提示信息的 xlsx 文件路径。")  
    parser.add_argument("--result_dir", type=str, default="/disk1/fjm/resultimg/unip/mis/test", help="保存生成图像的目录。")  
    args = parser.parse_args()  

    os.environ["HF_HOME"] = "/home/fujm/.cache/huggingface/hub"  

    # 解析设备列表  
    device_list = args.device.split(",")  
    device_ids = [int(dev.strip()) for dev in device_list]  
    num_devices = len(device_ids)  

    # 设置指定的 GPU 设备  
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, device_ids))  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    print(f"使用设备: {device}（物理GPU序号: {device_ids}）")  

    # 设置数据类型  
    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32  

    # 基础模型路径  
    base_model_path = "SG161222/Realistic_Vision_V5.1_noVAE"  
    vae_model_path = "stabilityai/sd-vae-ft-mse"  
    controlnet_pose_ckpt = "lllyasviel/control_v11p_sd15_openpose"  

    # 特定模型路径  
    image_encoder_path = "/disk1/fujm/IPAdapter/image_encoder/models/image_encoder"  
    ip_ckpt = "/disk1/fujm/IPAdapter/models/models/ip-adapter_sd15.bin"  
    face_backbone_ckpt = "/disk1/fujm/unip/UniPortrait/glint360k_curricular_face_r101_backbone.bin"  
    uniportrait_faceid_ckpt = "/disk1/fujm/unip/UniPortrait/uniportrait-faceid_sd15.bin"  
    uniportrait_router_ckpt = "/disk1/fujm/unip/UniPortrait/uniportrait-router_sd15.bin"

    # 加载 ControlNet 模型  
    pose_controlnet = ControlNetModel.from_pretrained(  
        controlnet_pose_ckpt,   
        torch_dtype=torch_dtype,   
        local_files_only=True  # 强制只从本地加载  
    )  

    # 加载 Stable Diffusion 管道  
    noise_scheduler = DDIMScheduler(  
        num_train_timesteps=1000,  
        beta_start=0.00085,  
        beta_end=0.012,  
        beta_schedule="scaled_linear",  
        clip_sample=False,  
        set_alpha_to_one=False,  
        steps_offset=1,  
    )  

    vae = AutoencoderKL.from_pretrained(  
        vae_model_path,   
        torch_dtype=torch_dtype,   
        local_files_only=True  # 强制只从本地加载  
    )  
    pipe = StableDiffusionControlNetPipeline.from_pretrained(  
        base_model_path,  
        controlnet=[pose_controlnet],  # 加[]是可能有多个控制模型的情况  
        torch_dtype=torch_dtype,  
        scheduler=noise_scheduler,  
        vae=vae,  
        local_files_only=True  # 强制只从本地加载  
    )  

    # 初始化 UniPortrait 管道  
    uniportrait_pipeline = UniPortraitPipeline(  
        pipe,  
        image_encoder_path,  
        ip_ckpt=ip_ckpt,  
        face_backbone_ckpt=face_backbone_ckpt,  
        uniportrait_faceid_ckpt=uniportrait_faceid_ckpt,  
        uniportrait_router_ckpt=uniportrait_router_ckpt,  
        device=device,  
        torch_dtype=torch_dtype  
    )  

    # 如果使用多个 GPU，启用 DataParallel  
    if num_devices > 1 and device.type == "cuda":  
        print(f"使用 {num_devices} 个 GPU 进行并行处理：{device_ids}")  
        uniportrait_pipeline.pipe = torch.nn.DataParallel(uniportrait_pipeline.pipe)  
    else:  
        print("使用单个 GPU 或 CPU。")  

    # 初始化面部检测应用  
    face_app = FaceAnalysis(  
        providers=['CUDAExecutionProvider' if device.type == "cuda" else 'CPUExecutionProvider'],  
        allowed_modules=["detection"]  
    )  
    face_app.prepare(ctx_id=0, det_size=(640, 640))  

    # 创建结果文件夹（如果不存在）  
    os.makedirs(args.result_dir, exist_ok=True)  

    # 读取 prompt.csv 文件并选择指定行范围  
    prompts = {}   
    df = pd.read_excel(args.prompt_xlsx, engine="openpyxl")
    # 注意：df默认带表头，index=0是数据区的第一行

    for idx in range(args.start_row, args.end_row + 1):
        # 行越界保护
        if idx >= len(df):
            break
        row = df.iloc[idx]
        try:
            # imagename = str(row[6])     
            imagename = str(row['image_name']).strip()
            # prompt_text = str(row[5])    
            prompt_text = str(row['prompt']).strip()
        except Exception:
            print(f"警告: 第 {idx+2} 行格式不正确，跳过。")
            continue
        try:
            numeric_part = int(os.path.splitext(imagename)[0])
            new_imagename = f"{numeric_part}.jpg"
        except Exception:
            new_imagename = imagename
        prompts[new_imagename] = prompt_text
    

    # 获取所有图像文件名（假设为 start_row 到 end_row 的序号）  
    image_filenames = [f"{i}.jpg" for i in range(args.start_row, args.end_row + 1)]  

    # 检查所有图像文件是否存在  
    for img_name in image_filenames:  
        img_path = os.path.join(args.image_dir, img_name)  
        if not os.path.isfile(img_path):  
            print(f"警告: {img_path} 不存在，跳过。")  

    # 遍历每个图像及其对应的提示  
    for img_name in tqdm(image_filenames, desc="生成图像"):  
        img_path = os.path.join(args.image_dir, img_name)  
        if not os.path.isfile(img_path):  
            continue  # 跳过不存在的图像  

        # 加载图像  
        pil_image = Image.open(img_path).convert("RGB")  

        # 获取对应的 prompt  
        prompt = prompts.get(img_name, "")  
        print(f"prompt is: '{prompt}'")
        # if not prompt:  
        #     print(f"警告: 没有找到 {img_name} 对应的 prompt，跳过。")  
        #     continue  

        # 调用生成函数  
        try:  
            generated_images = text_to_single_id_generation_process(  
                uniportrait_pipeline,  # 将 uniportrait_pipeline 作为参数传入  
                pil_faceid_image=pil_image,  
                pil_ip_image=pil_image,
                pil_faceid_supp_images=None,  # 如果有辅助图像，可以在此处传递  
                pil_faceid_mix_image_1=None,  
                mix_scale_1=0.0,  
                pil_faceid_mix_image_2=None,  
                mix_scale_2=0.0,  
                faceid_scale=args.faceid_scale,  
                face_structure_scale=args.face_structure_scale,  
                prompt=prompt,  
                negative_prompt="nsfw",  # 可以根据需要设置  
                num_samples=1,  
                seed=9,  
                image_resolution="512x512",  
                inference_steps=30,  
                face_app=face_app  
            )  

            # 保存生成的图像  
            for idx, gen_img in enumerate(generated_images):  
                if isinstance(gen_img, Image.Image):  
                    # 定义保存路径  
                    save_path = os.path.join(args.result_dir, f"{os.path.splitext(img_name)[0]}_gen_{idx}_{prompt}.jpg")  
                    gen_img.save(save_path)  
        except Exception as e:  
            print(f"错误: 处理 {img_name} 时出错: {e}")  

    print("所有图像生成完毕。")  
