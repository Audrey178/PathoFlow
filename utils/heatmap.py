import cv2
import numpy as np
import os
import glob
from tqdm import tqdm

# --- CONFIGURATION ---
STITCHED_MAP_PATH = "/mnt/disk4/video-panninng-classification/stitched/Adenoma/24b0848.jpeg"   
FRAMES_FOLDER = "/mnt/disk4/video-panninng-classification/datasets/frames_to_find_heatmap/24b0848_frames"             
INDICES_TO_PROCESS = list(range(len(os.listdir(FRAMES_FOLDER))))  # Process all frames
INDICES_TO_PROCESS = [0, 30, 43, 95, 175, 193, 262, 334, 440]
# INDICES_TO_PROCESS = INDICES_TO_PROCESS[::20]  # Example: process every 10th frame

print(INDICES_TO_PROCESS)
OUTPUT_FILENAME = "/mnt/disk4/video-panninng-classification/heatmap_outputs/24b0848_OF.jpg"
# VISUALIZATION
OVERLAY_OPACITY = 0.6 
COLORMAP = cv2.COLORMAP_JET 

# SMOOTHING SETTINGS
# Sigma = How much to blur. Higher = Smoother "hills", fewer jagged edges.
# Kernel Size must be odd (e.g., 101, 201).
SMOOTH_SIGMA = 50 
SMOOTH_KERNEL = (101, 101)

# OPTIMIZATION
DETECTION_WIDTH = 5000 
MIN_BRIGHTNESS = 40
MIN_SATURATION = 20

# Force CPU to avoid OpenCL crash
cv2.ocl.setUseOpenCL(False)
# ---------------------

def crop_black_borders(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        if w * h > 0.2 * (img.shape[0] * img.shape[1]):
            return img[y:y+h, x:x+w]
    return img

def create_tissue_mask(img):
    print("  -> Creating tissue mask...")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    mask = (s > MIN_SATURATION) & (v > MIN_BRIGHTNESS)
    mask = mask.astype(np.uint8) * 255
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def generate_smooth_heatmap():
    print("--- Step 3: Smooth Heatmap Generation ---")
    
    if not os.path.exists(STITCHED_MAP_PATH):
        print(f"Error: {STITCHED_MAP_PATH} missing.")
        return
        
    print("Loading WSI...")
    full_map = cv2.imread(STITCHED_MAP_PATH)
    orig_h, orig_w = full_map.shape[:2]
    
    # 1. Generate Tissue Mask
    tissue_mask = create_tissue_mask(full_map)

    # 2. Setup SIFT Proxy
    scale_factor = DETECTION_WIDTH / float(orig_w)
    if scale_factor > 1.0: scale_factor = 1.0
    new_h = int(orig_h * scale_factor)
    proxy_map = cv2.resize(full_map, (int(orig_w * scale_factor), new_h))
    
    sift = cv2.SIFT_create(contrastThreshold=0.01, edgeThreshold=20)
    kp_map, des_map = sift.detectAndCompute(cv2.cvtColor(proxy_map, cv2.COLOR_BGR2GRAY), None)
    
    if des_map is None: 
        print("Error: No features in map.")
        return
    bf = cv2.BFMatcher()
    
    # 3. Gather Frames
    all_files = sorted(glob.glob(os.path.join(FRAMES_FOLDER, '*')))
    valid = ('.jpg', '.png', '.tif')
    all_files = [f for f in all_files if f.lower().endswith(valid)]
    
    selected_paths = []
    for idx in INDICES_TO_PROCESS:
        if 0 <= idx < len(all_files): selected_paths.append(all_files[idx])
            
    # 4. Accumulate Counts (The "Raw" Data)
    heatmap_accumulator = np.zeros((orig_h, orig_w), dtype=np.float32)
    matches_found = 0
    
    print(f"Retracking {len(selected_paths)} frames...")
    for path in tqdm(selected_paths):
        original = cv2.imread(path)
        if original is None: continue
        
        cropped = crop_black_borders(original)
        kp_fr, des_fr = sift.detectAndCompute(cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY), None)
        
        if des_fr is None or len(des_fr) < 5: continue
        
        matches = bf.knnMatch(des_fr, des_map, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        
        if len(good) >= 4:
            src = np.float32([kp_fr[m.queryIdx].pt for m in good]).reshape(-1,1,2)
            dst = np.float32([kp_map[m.trainIdx].pt for m in good]).reshape(-1,1,2)
            M, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            
            if M is not None:
                h_fr, w_fr = cropped.shape[:2]
                pts = np.float32([[0,0], [0,h_fr], [w_fr,h_fr], [w_fr,0]]).reshape(-1,1,2)
                dst_full = cv2.perspectiveTransform(pts, M) / scale_factor
                x, y, w, h = cv2.boundingRect(np.int32(dst_full))
                
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(orig_w, x + w), min(orig_h, y + h)
                
                if x2 > x1 and y2 > y1:
                    # Add 1.0 to the box region
                    heatmap_accumulator[y1:y2, x1:x2] += 1.0
                    matches_found += 1

    if matches_found == 0:
        print("No matches found.")
        return

    # --- 5. THE SMOOTHING STEP ---
    print("Smoothing heatmap data...")
    # This blurs the raw counts (float32), turning blocks into hills.
    heatmap_smooth = cv2.GaussianBlur(heatmap_accumulator, SMOOTH_KERNEL, SMOOTH_SIGMA)

    # 6. Normalize & Colorize
    max_val = np.max(heatmap_smooth)
    if max_val == 0: max_val = 1.0
    
    print(f"Max density (smoothed): {max_val:.2f}")
    
    heatmap_norm = heatmap_smooth / max_val
    heatmap_uint8 = (heatmap_norm * 255).astype(np.uint8)
    
    # Apply Color Map (Jet: Blue -> Red)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, COLORMAP)

    # 7. Apply Tissue Mask (Cut out background)
    heatmap_masked = cv2.bitwise_and(heatmap_color, heatmap_color, mask=tissue_mask)
    
    # 8. Overlay
    print("Blending...")
    output_img = full_map.copy()
    img_roi = cv2.bitwise_and(output_img, output_img, mask=tissue_mask)
    
    blended_roi = cv2.addWeighted(img_roi, 1.0 - OVERLAY_OPACITY, 
                                  heatmap_masked, OVERLAY_OPACITY, 0)
    
    mask_indices = np.where(tissue_mask > 0)
    output_img[mask_indices] = blended_roi[mask_indices]

    print(f"Saving to {OUTPUT_FILENAME}...")
    cv2.imwrite(OUTPUT_FILENAME, output_img)
    print("Done.")

if __name__ == "__main__":
    generate_smooth_heatmap()