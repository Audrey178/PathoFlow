import cv2
import numpy as np


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
    
    
def get_smart_indices_from_frames(frames):
    """
    Processes a list of frames (numpy arrays or PIL Images) and selects 
    indices based on optical flow motion thresholds.
    """
    MIN_FEATURES = 20
    MIN_MOTION_PX = 2
    MAX_JITTER_PX = 3
    COS_TURN_THRESH = 0.3
    OVERLAP_RATIO = 0.3
    
    INSPECTION = "SLOW_SWEEP"
    RASTER = "NORMAL_RASTER"
    
    overlap_ratio = OVERLAP_RATIO

    if not frames:
        print("Error: Input frame list is empty.")
        return []

    # --- Helper: Ensure frame is OpenCV compatible (Numpy BGR) ---
    def to_cv2_frame(frame):
        # If it's a PIL Image (checks for 'save' attribute usually, or specifically the lack of 'shape')
        if not hasattr(frame, 'shape'): 
            # Convert PIL to Numpy (RGB)
            arr = np.array(frame)
            # Convert RGB to BGR (standard OpenCV format)
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return frame

    # Prepare first frame
    prev_frame = to_cv2_frame(frames[0])
    
    if prev_frame is None:
        raise RuntimeError("First frame in list is None/Invalid.")

    h, w = prev_frame.shape[:2]
    BASE_DISTANCE = w * (1.0 - overlap_ratio)
    
    # ROI selection
    ry1, ry2 = int(0.2*h), int(0.8*h)
    rx1, rx2 = int(0.2*w), int(0.8*w)

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_roi = prev_gray[ry1:ry2, rx1:rx2]

    selected_indices = [0]
    
    acc_x = acc_y = 0.0
    acc_dir = None
    speed_hist = []

    sequence_mode = RASTER
    sequence_stitch_indices = []

    for i in range(1, len(frames)):
        # Convert current frame if necessary
        curr_frame = to_cv2_frame(frames[i])

        if curr_frame is None:
            continue

        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        curr_roi = curr_gray[ry1:ry2, rx1:rx2]

        pts_prev = cv2.goodFeaturesToTrack(prev_roi, 300, 0.01, 8)
        
        if pts_prev is None or len(pts_prev) < MIN_FEATURES:
            prev_roi = curr_roi
            continue

        pts_curr, status, _ = cv2.calcOpticalFlowPyrLK(prev_roi, curr_roi, pts_prev, None)
        
        status = status.reshape(-1)
        pts_prev = pts_prev.reshape(-1, 2)
        pts_curr = pts_curr.reshape(-1, 2)
        valid = status == 1
        
        if valid.sum() < MIN_FEATURES:
            prev_roi = curr_roi
            continue

        flow = pts_curr[valid] - pts_prev[valid]
        dx, dy = flow[:, 0], flow[:, 1]

        step_dx = np.median(dx)
        step_dy = np.median(dy)
        speed = np.hypot(step_dx, step_dy)
        jitter = np.median(np.abs(np.hypot(dx, dy) - speed))

        if speed < MIN_MOTION_PX or jitter > MAX_JITTER_PX:
            prev_roi = curr_roi
            continue

        speed_hist.append(speed)
        if len(speed_hist) > 30:
            speed_hist.pop(0)
        median_speed = np.median(speed_hist)

        curr_dir = np.array([step_dx, step_dy])
        curr_dir /= (np.linalg.norm(curr_dir) + 1e-6)

        # Ensure determine_mode exists in your context
        mode = determine_mode(speed, median_speed)

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
            selected_indices.append(i)
            acc_x = acc_y = 0.0
            acc_dir = None

            if sequence_mode == mode:
                sequence_stitch_indices.append(i)
            else:
                sequence_mode = mode
                sequence_stitch_indices = [i]
        else:
            acc_dir = curr_dir if acc_dir is None else (acc_dir + curr_dir) / (np.linalg.norm(acc_dir + curr_dir) + 1e-6)

        prev_roi = curr_roi

    last_idx = len(frames) - 1
    if selected_indices[-1] != last_idx:
        selected_indices.append(last_idx)
        
    return selected_indices