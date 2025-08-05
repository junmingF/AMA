
import os
import argparse
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm

# Import the actual PixtralContentFilter from your installed package
from flux.content_filters import PixtralContentFilter

def process_images_in_folder(image_folder, device_str):
    """
    Batch process all images in a folder and detect safety issues using PixtralContentFilter.
    """
    # --- 1. Initialize model ---
    try:
        device = torch.device(device_str)
        print(f"Running on device: {device}")
        integrity_checker = PixtralContentFilter(device)
    except NameError:
        print("\nError: 'PixtralContentFilter' is not defined.")
        print("Please make sure you have installed the library containing this filter and can import it successfully.")
        return
    except Exception as e:
        print(f"Error initializing model: {e}")
        return

    # --- 2. Prepare file list ---
    try:
        all_files = os.listdir(image_folder)
        image_extensions = ['.png', '.jpg', '.jpeg', '.webp', '.bmp']
        image_files = [f for f in all_files if os.path.splitext(f)[1].lower() in image_extensions]
        if not image_files:
            print(f"Error: No supported image files found in folder '{image_folder}'.")
            return
    except FileNotFoundError:
        print(f"Error: Folder does not exist: '{image_folder}'")
        return
    except Exception as e:
        print(f"Error reading folder: {e}")
        return

    # Statistics counters
    safe_count = 0
    unsafe_count = 0
    skipped_count = 0

    print(f"Found {len(image_files)} images, starting detection...")

    # --- 3. Process each image ---
    for filename in tqdm(image_files, desc="Detecting images"):
        image_path = os.path.join(image_folder, filename)

        try:
            # 1. Load image
            image = Image.open(image_path).convert("RGB")

            # 2. Preprocess image
            image_ = np.array(image) / 255.0
            image_ = 2 * image_ - 1
            image_ = torch.from_numpy(image_).to(device, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)

            # 3. Run detection
            if integrity_checker.test_image(image_):
                # Unsafe content detected
                unsafe_count += 1
                tqdm.write(f"❌  Unsafe: {filename}")
            else:
                # Image is safe
                safe_count += 1

        except Exception as e:
            # If image is corrupted or fails, skip it
            tqdm.write(f"⚠️  Skipped (processing failed): {filename}, reason: {e}")
            skipped_count += 1
            continue

    # --- 4. Output summary statistics ---
    total_processed = safe_count + unsafe_count
    print("\n" + "="*30)
    print("           Detection Summary")
    print("="*30)
    print(f"Total images detected:    {total_processed}")
    print(f"✅ Safe images:           {safe_count}")
    print(f"❌ Unsafe images:         {unsafe_count}")
    print(f"⏭️ Skipped (error):       {skipped_count}")
    print("="*30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Batch detect image content safety in a folder using PixtralContentFilter.")
    parser.add_argument('--folder', type=str, default='/disk1/fjm/test500', help="Path to the folder containing images to detect.")
    parser.add_argument('--device', type=str, default='cuda:0', help="Device for running the model, e.g., 'cuda:0' or 'cpu'.")

    args = parser.parse_args()

    process_images_in_folder(args.folder, args.device)
