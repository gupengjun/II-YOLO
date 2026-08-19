import cv2
import numpy as np
import os

def enhance_water(img, water_mask, dark_gain=1.6, bright_gain=1.15, noise_std=10):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    mask = water_mask.copy()
    if mask.ndim == 3:
        mask = mask[:, :, 0] 
    if mask.max() <= 1.0:
        mask = (mask * 255).astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)
    if mask.shape[0] != h or mask.shape[1] != w:
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    mask_float = mask.astype(np.float32) / 255.0
    mask_float = cv2.GaussianBlur(mask_float, (41, 41), 0)
    mask_3d = mask_float[:, :, np.newaxis]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    V = hsv[:, :, 2]
    dark_region = (V < 120).astype(np.float32)
    bright_region = 1.0 - dark_region
    dark_region = cv2.GaussianBlur(dark_region, (51, 51), 0)
    bright_region = cv2.GaussianBlur(bright_region, (51, 51), 0)
    V_new = V * dark_gain * dark_region + V * bright_gain * bright_region
    V_new = np.clip(V_new, 0, 255)
    hsv[:, :, 2] = V_new
    enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    noise = np.random.normal(0, noise_std, (h, w, 3)).astype(np.float32)
    kernel = cv2.getGaussianKernel(21, 5)
    kernel = kernel @ kernel.T
    for c in range(3):
        noise[:, :, c] = cv2.filter2D(noise[:, :, c], -1, kernel)
    specular = enhanced.astype(np.float32) + noise * mask_3d
    specular = np.clip(specular, 0, 255).astype(np.uint8)
    out = img.astype(np.float32) * (1 - mask_3d) + specular.astype(np.float32) * mask_3d
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out
def process(image_path, mask_path, out_path):
    img = cv2.imread(image_path)
    mask = cv2.imread(mask_path, 0)  # 灰度 mask

    if img is None:
        print("Image not found:", image_path)
        return
    if mask is None:
        print("Mask not found:", mask_path)
        return

    print(f"Processing: {os.path.basename(image_path)}")
    print(f"Image shape: {img.shape}, Mask shape: {mask.shape}")

    enhanced = enhance_water(img, mask)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, enhanced)
    print("Saved →", out_path)

def main():
    img_dir = "out/test/images"
    mask_dir = "out/test/mask"
    out_dir = "out/test/enhance"
    if not os.path.exists(img_dir):
        print(f"Image directory not found: {img_dir}")
        return
    if not os.path.exists(mask_dir):
        print(f"Mask directory not found: {mask_dir}")
        return
    os.makedirs(out_dir, exist_ok=True)

    image_files = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    if not image_files:
        print("No images found in", img_dir)
        return

    print(f"Found {len(image_files)} images to process")

    for name in image_files:
        img_path = os.path.join(img_dir, name)
        mask_path = os.path.join(mask_dir, name)
        out_path = os.path.join(out_dir, name)

        if not os.path.exists(mask_path):
            print("No mask:", mask_path)
            continue

        process(img_path, mask_path, out_path)

    print("Processing completed!")


if __name__ == "__main__":
    main()