import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

VIDEO_DIR = "new_data/BS Cuong"
FRAME_DIR = "datasets/frames_v3"
SCENE_THRESH = 0.05
MAX_WORKERS = 4

label2label = {
    "A": "Adenoma",
    "N": "Normal",
    "M": "Malignant"
}

def extract_frames(video_path, output_dir):
    name = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.path.join(output_dir, name)
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vsync", "0",
        f"{out_dir}/frame_%06d.png"
    ]
    subprocess.run(cmd)
    return f"[OK] {name}"

if __name__ == "__main__":
    
    dirs = os.listdir(VIDEO_DIR)
    
    for dir in dirs:
        print(f"=============={dir} processing===============")
        input_dir = os.path.join(VIDEO_DIR, dir)
        label = label2label[dir.split("-")[0]]
        output_dir = os.path.join(FRAME_DIR, label)
        os.makedirs(output_dir, exist_ok=True)
        videos = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
              if f.lower().endswith((".mp4", ".avi", ".mkv"))]

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            for result in ex.map(extract_frames, videos, [output_dir]*len(videos)):
                print(result)
