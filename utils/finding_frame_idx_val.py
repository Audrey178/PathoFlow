import cv2
import numpy as np
import os
import re
import csv
import shutil
from tempfile import NamedTemporaryFile
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


def determine_mode(speed, median_speed):
    INSPECTION = "SLOW_SWEEP"
    FAST_SWEEP = "FAST_SWEEP"
    RASTER = "NORMAL_RASTER"
    
    if speed < 0.6 * median_speed:
        return INSPECTION
    elif speed > 1.5 * median_speed:
        return FAST_SWEEP
    else:
        return RASTER

def get_smart_indices_from_folder(folder_path):
    # =====================
    # PARAMETERS
    # =====================
    #[40, 4, 6, 0.3, 0.2], [20, 2, 3, 0.3, 0.4], [20, 2, 3, 0.3, 0.5], [10, 2, 3, 0.3, 0.5], [15, 2, 3, 0.3, 0.3]
    MIN_FEATURES = 50       # High texture expectation for tissue
    MIN_MOTION_PX = 2     # Allow very slow, deliberate scanning
    MAX_JITTER_PX = 3       # 2D flat plane = rigid motion; reject wobbles

    # Sweep / sampling
    COS_TURN_THRESH = 0.3   # Detect turns in the raster pattern earlier
    OVERLAP_RATIO = 0.2    # High overlap for robust stitching of repetitive cells

    # Modes
    INSPECTION = "SLOW_SWEEP"
    FAST_SWEEP = "FAST_SWEEP"
    RASTER = "NORMAL_RASTER"
    overlap_ratio = OVERLAP_RATIO
    
    # 1. Get and sort file names numerically
    # Expects filenames like "0.jpg", "1.jpg" or "frame_0.png", etc.
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.avi'} # added .avi just in case, though usually folders have imgs
    files = sorted(
        [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in valid_exts],
        key=lambda f: int(re.sub(r'\D', '', f))
    )

    if not files:
        print("Error: No images found in folder.")
        return []

    # 2. Initialize with the first frame
    first_frame_path = os.path.join(folder_path, files[0])
    prev_frame = cv2.imread(first_frame_path)
    if prev_frame is None:
        raise RuntimeError(f"Cannot read first frame: {first_frame_path}")

    h, w = prev_frame.shape[:2]
    BASE_DISTANCE = w * (1.0 - overlap_ratio)

    # ROI for Optical Flow (Center 60%)
    ry1, ry2 = int(0.2*h), int(0.8*h)
    rx1, rx2 = int(0.2*w), int(0.8*w)

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_roi = prev_gray[ry1:ry2, rx1:rx2]

    # Initialize tracking variables
    # We map the file index in the sorted list to the frame order
    selected_indices = [0] # Always keep the first frame
    
    acc_x = acc_y = 0.0
    acc_dir = None
    speed_hist = []

    sequence_mode = RASTER
    sequence_stitch_indices = []

    # print(f"=== Processing {len(files)} frames from: {folder_path} ===")

    # 3. Iterate starting from the second frame
    for i in range(1, len(files)):
        filename = files[i]
        curr_frame_path = os.path.join(folder_path, filename)
        curr_frame = cv2.imread(curr_frame_path)

        if curr_frame is None:
            continue

        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        curr_roi = curr_gray[ry1:ry2, rx1:rx2]

        # --- Optical Flow Calculation ---
        pts_prev = cv2.goodFeaturesToTrack(prev_roi, 300, 0.01, 8)
        
        # If not enough features, treat as static/invalid and skip update
        if pts_prev is None or len(pts_prev) < MIN_FEATURES:
            prev_roi = curr_roi
            # Clean up memory
            del curr_frame
            continue

        pts_curr, status, _ = cv2.calcOpticalFlowPyrLK(prev_roi, curr_roi, pts_prev, None)
        
        # Filter valid points
        status = status.reshape(-1)
        pts_prev = pts_prev.reshape(-1, 2)
        pts_curr = pts_curr.reshape(-1, 2)
        valid = status == 1
        
        if valid.sum() < MIN_FEATURES:
            prev_roi = curr_roi
            del curr_frame
            continue

        flow = pts_curr[valid] - pts_prev[valid]
        dx, dy = flow[:, 0], flow[:, 1]

        step_dx = np.median(dx)
        step_dy = np.median(dy)
        speed = np.hypot(step_dx, step_dy)
        jitter = np.median(np.abs(np.hypot(dx, dy) - speed))

        # --- Filter: Low Motion or High Jitter ---
        if speed < MIN_MOTION_PX or jitter > MAX_JITTER_PX:
            prev_roi = curr_roi
            del curr_frame
            continue

        # Update History
        speed_hist.append(speed)
        if len(speed_hist) > 30:
            speed_hist.pop(0)
        median_speed = np.median(speed_hist)

        curr_dir = np.array([step_dx, step_dy])
        curr_dir /= (np.linalg.norm(curr_dir) + 1e-6)

        # --- Logic: Determine Mode ---
        mode = determine_mode(speed, median_speed)

        # --- Logic: Sampling / Trigger ---
        acc_x += step_dx
        acc_y += step_dy
        acc_dist = np.hypot(acc_x, acc_y)

        trigger = False
        
        if mode == INSPECTION:
            if acc_dist > 0.5 * BASE_DISTANCE:
                trigger = True
        else:
            if acc_dist >= BASE_DISTANCE:
                trigger = True
            elif acc_dir is not None and np.dot(curr_dir, acc_dir) < COS_TURN_THRESH:
                trigger = True

        if trigger:
            # We select the current index 'i' (which corresponds to files[i])
            selected_indices.append(i)
            
            acc_x = acc_y = 0.0
            acc_dir = None

            # Logging logic (Optional, helps verify behavior)
            if sequence_mode == mode:
                sequence_stitch_indices.append(i)
            else:
                # if sequence_stitch_indices:
                #     print(f"[{sequence_mode:<12}] frames {sequence_stitch_indices[0]:4d} -> {sequence_stitch_indices[-1]:4d}")
                sequence_mode = mode
                sequence_stitch_indices = [i]
        else:
            # Accumulate direction
            acc_dir = curr_dir if acc_dir is None else (acc_dir + curr_dir) / (np.linalg.norm(acc_dir + curr_dir) + 1e-6)

        prev_roi = curr_roi
        del curr_frame # Explicit memory free

    # Log last sequence
    # if sequence_stitch_indices:
    #     print(f"[{sequence_mode:<12}] frames {sequence_stitch_indices[0]:4d} -> {sequence_stitch_indices[-1]:4d}")

    # Force last frame inclusion if not already added
    last_idx = len(files) - 1
    if selected_indices[-1] != last_idx:
        selected_indices.append(last_idx)

    # print(f"Total frames selected: {len(selected_indices)}")
    return selected_indices


def process_row(row):
    """
    This function runs on a separate CPU core.
    It takes a single row, does the math, and returns the modified row.
    """
    output_column = 'selected_frames'
    
    frames_folder = row.get('path', '')
    # Your logic to adjust the path
    frames_folder = frames_folder[:-4].replace('videos', 'frames')
    
    if frames_folder:
        try:
            # Call your algorithm (Ensure this function is defined above!)
            indices = get_smart_indices_from_folder(frames_folder)
            row[output_column] = indices[:]
        except Exception as e:
            # Use a safe fallback if the function fails
            print(f"Error on {frames_folder}: {e}")
            row[output_column] = "[]"
    else:
        row[output_column] = "[]"

    return row

# --- 2. MAIN EXECUTION ---
if __name__ == "__main__":
    folder_in = "/mnt/disk4/video-panninng-classification/datasets/frames"
    csv_path = '/mnt/disk4/video-panninng-classification/datasets/csv/val.csv'
    full_path = csv_path
    
    print(f"Processing CSV: {full_path}")

    # Prepare the temporary file
    temp_file = NamedTemporaryFile(mode='w', delete=False, newline='', encoding='utf-8')

    with open(full_path, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        
        # Add new column name if it doesn't exist
        output_column = 'selected_frames'
        if output_column not in fieldnames:
            fieldnames.append(output_column)
        
        # Read all rows into RAM first
        rows = list(reader)

    # --- Start Parallel Processing ---
    # Python will use all available CPU cores by default
    print(f"Starting parallel processing on {len(rows)} rows...")
    
    processed_rows = []
    with ProcessPoolExecutor() as executor:
        # map() distributes the rows across cores and keeps the order correct
        results = executor.map(process_row, rows)
        
        # Wrap results in tqdm for the progress bar
        processed_rows = list(tqdm(results, total=len(rows), desc="Processing videos", unit="row"))

    # --- Write Results ---
    with temp_file:
        writer = csv.DictWriter(temp_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(processed_rows)

    # Replace original file with the new temp file
    shutil.move(temp_file.name, full_path)
    print(f"Finished updating {full_path}\n")