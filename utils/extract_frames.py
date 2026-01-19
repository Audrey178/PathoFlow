import cv2
import os

# --- CONFIGURATION ---
VIDEO_PATH = "/mnt/disk4/video-panninng-classification/datasets/videos/Adenoma/25a6028.avi"
OUTPUT_FOLDER = "/mnt/disk4/video-panninng-classification/datasets/frames_to_find_heatmap/25a6028_frames"
# ---------------------

def extract_frames():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        
    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video found: {total_frames} frames.")
    print(f"Extracting to '{OUTPUT_FOLDER}'...")
    
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Save filename with padding (e.g., frame_000150.jpg) for easy sorting
        filename = os.path.join(OUTPUT_FOLDER, f"frame_{idx:06d}.jpg")
        
        # JPEG quality 90 is a good balance for pathology
        cv2.imwrite(filename, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        
        if idx % 100 == 0:
            print(f"Extracted {idx}/{total_frames}")
        
        idx += 1
        
    cap.release()
    print("Extraction complete.")

if __name__ == "__main__":
    extract_frames()