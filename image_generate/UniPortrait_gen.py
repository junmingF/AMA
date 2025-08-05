import os  
import pandas as pd 
import argparse  
from io import BytesIO  
import cv2  
import numpy as np  
import torch  
from PIL import Image  
from tqdm import tqdm  

# Import previously defined functions  
from uniportrait import inversion  
from uniportrait.uniportrait_attention_processor import attn_args  
from uniportrait.uniportrait_pipeline import UniPortraitPipeline  
from insightface.app import FaceAnalysis  
from insightface.utils import face_align  
from diffusers import DDIMScheduler, AutoencoderKL, ControlNetModel, StableDiffusionControlNetPipeline  

# Function to pad an image
def pad_np_bgr_image(np_image, scale=1.25):  
    assert scale >= 1.0, "scale should be >= 1.0"  
    pad_scale = scale - 1.0  
    h, w = np_image.shape[:2]  
    top = bottom = int(h * pad_scale)  
    left = right = int(w * pad_scale)  
    ret = cv2.copyMakeBorder(np_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(128, 128, 128))  
    return ret, (left, top)  # Return padded image and top-left offset  

# Function to process and align face images  
def process_faceid_image(pil_faceid_image, face_app):  
    np_faceid_image = np.array(pil_faceid_image.convert("RGB"))  
    img = cv2.cvtColor(np_faceid_image, cv2.COLOR_RGB2BGR)  
    faces = face_app.get(img)  # bgr  
    if len(faces) == 0:  
        # Try with padding  
        _h, _w = img.shape[:2]  
        _img, left_top_coord = pad_np_bgr_image(img)  
        faces = face_app.get(_img)  
        if len(faces) == 0:  
            print("Warning: No face detected in the image. Skipping...")  
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

# Prepare conditional parameters for a single face-id task
def prepare_single_faceid_cond_kwargs(pil_faceid_image=None, pil_faceid_supp_images=None,  
                                      pil_faceid_mix_images=None, mix_scales=None, face_app=None):  
    pil_faceid_align_images = []  
    # Process main face image  
    if pil_faceid_image:  
        processed_image = process_faceid_image(pil_faceid_image, face_app)  
        if processed_image:  
            pil_faceid_align_images.append(processed_image)  
    # Process supplementary face images  
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
    # Process mix images and corresponding scales  
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

# Generate images for a single identity using UniPortraitPipeline  
def text_to_single_id_generation_process(  
        uniportrait_pipeline,
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

    # Reset attention parameters  
    attn_args.reset()  
    # Set face conditioning parameters  
    attn_args.lora_scale = 1.0 if len(cond_faceids) == 1 else 0.0  
    attn_args.multi_id_lora_scale = 1.0 if len(cond_faceids) > 1 else 0.0  
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

    parser = argparse.ArgumentParser(description="Script for batch image generation using UniPortrait.")  
    parser.add_argument("--device", type=str, default="2", help="GPU device to use, e.g. '0' or '0,1'.")  
    parser.add_argument("--start_row", type=int, default=0, help="Start row (zero-based) in the XLSX to process.")  
    parser.add_argument("--end_row", type=int, default=10, help="End row (inclusive) in the XLSX to process.")  
    parser.add_argument("--faceid_scale", type=float, default=0.8, help="Higher values: more like ID photo face.")  
    parser.add_argument("--face_structure_scale", type=float, default=0.4, help="Higher values: more like reference photo pose.")  
    parser.add_argument("--image_dir", type=str, default="/disk1/fjm/img", help="Directory containing input images.") 
    parser.add_argument("--prompt_xlsx", type=str, default="/disk1/fjm/LLM/new_prompt/output/full_0803_0-2000.xlsx", help="Path to XLSX prompt file.")  
    parser.add_argument("--result_dir", type=str, default="/disk1/fjm/resultimg/unip/mis/test", help="Output directory to save results.")  
    args = parser.parse_args()  

    os.environ["HF_HOME"] = "/home/fujm/.cache/huggingface/hub"  

    # Parse device list  
    device_list = args.device.split(",")  
    device_ids = [int(dev.strip()) for dev in device_list]  
    num_devices = len(device_ids)  

    # Set CUDA_VISIBLE_DEVICES  
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, device_ids))  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    print(f"Using device: {device} (physical GPU IDs: {device_ids})")  

    # Set torch dtype  
    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32  

    # Base model paths  
    base_model_path = "SG161222/Realistic_Vision_V5.1_noVAE"  
    vae_model_path = "stabilityai/sd-vae-ft-mse"  
    controlnet_pose_ckpt = "lllyasviel/control_v11p_sd15_openpose"  

    # Specific model paths  
    image_encoder_path = "/disk1/fujm/IPAdapter/image_encoder/models/image_encoder"  
    ip_ckpt = "/disk1/fujm/IPAdapter/models/models/ip-adapter_sd15.bin"  
    face_backbone_ckpt = "/disk1/fujm/unip/UniPortrait/glint360k_curricular_face_r101_backbone.bin"  
    uniportrait_faceid_ckpt = "/disk1/fujm/unip/UniPortrait/uniportrait-faceid_sd15.bin"  
    uniportrait_router_ckpt = "/disk1/fujm/unip/UniPortrait/uniportrait-router_sd15.bin"

    # Load ControlNet model  
    pose_controlnet = ControlNetModel.from_pretrained(  
        controlnet_pose_ckpt,   
        torch_dtype=torch_dtype,   
        local_files_only=True  
    )  

    # Load Stable Diffusion pipeline  
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
        local_files_only=True  
    )  
    pipe = StableDiffusionControlNetPipeline.from_pretrained(  
        base_model_path,  
        controlnet=[pose_controlnet], 
        torch_dtype=torch_dtype,  
        scheduler=noise_scheduler,  
        vae=vae,  
        local_files_only=True  
    )  

    # Initialize UniPortrait pipeline  
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

    # Enable DataParallel if using multiple GPUs  
    if num_devices > 1 and device.type == "cuda":  
        print(f"Using {num_devices} GPUs: {device_ids} for parallel execution")  
        uniportrait_pipeline.pipe = torch.nn.DataParallel(uniportrait_pipeline.pipe)  
    else:  
        print("Using single GPU or CPU.")  

    # Initialize face detection app  
    face_app = FaceAnalysis(  
        providers=['CUDAExecutionProvider' if device.type == "cuda" else 'CPUExecutionProvider'],  
        allowed_modules=["detection"]  
    )  
    face_app.prepare(ctx_id=0, det_size=(640, 640))  

    # Create result directory if not exists  
    os.makedirs(args.result_dir, exist_ok=True)  

    # Read prompt XLSX and select the specified row range  
    prompts = {}   
    df = pd.read_excel(args.prompt_xlsx, engine="openpyxl")
    # Note: df includes header by default, index=0 is first data row

    for idx in range(args.start_row, args.end_row + 1):
        # Out-of-bounds protection
        if idx >= len(df):
            break
        row = df.iloc[idx]
        try:
            imagename = str(row['image_name']).strip()
            prompt_text = str(row['prompt']).strip()
        except Exception:
            print(f"Warning: Bad format at row {idx+2}, skipped.")
            continue
        try:
            numeric_part = int(os.path.splitext(imagename)[0])
            new_imagename = f"{numeric_part}.jpg"
        except Exception:
            new_imagename = imagename
        prompts[new_imagename] = prompt_text
    

    # Assume image file names are "{idx}.jpg" for start_row ~ end_row  
    image_filenames = [f"{i}.jpg" for i in range(args.start_row, args.end_row + 1)]  

    # Check all image files exist  
    for img_name in image_filenames:  
        img_path = os.path.join(args.image_dir, img_name)  
        if not os.path.isfile(img_path):  
            print(f"Warning: {img_path} does not exist, skipping.")  

    # Iterate images and corresponding prompts  
    for img_name in tqdm(image_filenames, desc="Generating images"):  
        img_path = os.path.join(args.image_dir, img_name)  
        if not os.path.isfile(img_path):  
            continue  # Skip non-existent images  

        # Load image  
        pil_image = Image.open(img_path).convert("RGB")  

        # Get prompt  
        prompt = prompts.get(img_name, "")  
        print(f"prompt is: '{prompt}'")

        # Call the generation function  
        try:  
            generated_images = text_to_single_id_generation_process(  
                uniportrait_pipeline,
                pil_faceid_image=pil_image,  
                pil_ip_image=pil_image,
                pil_faceid_supp_images=None,
                pil_faceid_mix_image_1=None,  
                mix_scale_1=0.0,  
                pil_faceid_mix_image_2=None,  
                mix_scale_2=0.0,  
                faceid_scale=args.faceid_scale,  
                face_structure_scale=args.face_structure_scale,  
                prompt=prompt,  
                negative_prompt="nsfw",  
                num_samples=1,  
                seed=9,  
                image_resolution="512x512",  
                inference_steps=30,  
                face_app=face_app  
            )  

            # Save generated images  
            for idx, gen_img in enumerate(generated_images):  
                if isinstance(gen_img, Image.Image):  
                    save_path = os.path.join(args.result_dir, f"{os.path.splitext(img_name)[0]}_gen_{idx}_{prompt}.jpg")  
                    gen_img.save(save_path)  
        except Exception as e:  
            print(f"Error: Error occurred while processing {img_name}: {e}")  

    print("All images have been generated.")