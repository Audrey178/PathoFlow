import torch
import argparse
import os
import sys
import cv2
import timm
import numpy as np
from PIL import Image
from torchvision import transforms as T

# --- 1. SETUP IMPORTS ---
# Add current directory to path to ensure local modules are found
sys.path.append(os.getcwd())

try:
    from utils.feat_extract import TimmViTEncoder
    from models.vit_transformer_model import VTransAdaptive
    from utils.finding_frame_idx_train import get_smart_indices_from_folder
except ImportError as e:
    sys.exit(f"\n[Error] Import failed: {e}.\nEnsure you are running this script from the project root.\n")

def get_args():
    parser = argparse.ArgumentParser(description="Inference for Single Video")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input .mp4 file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained .pth checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # Model Config (Defaults match your training settings)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--hidden_size", type=int, default=1536, help="Embedding dim (1536 for UNI2-h)")
    parser.add_argument("--ratio", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--n_masked_patch", type=int, default=10)
    parser.add_argument("--mask_drop", type=float, default=0.1)
    
    return parser.parse_args()

def load_video_tensor(video_path):
    """
    Reads video -> Smart Sample (using your imported util) -> ToTensor
    """
    print(f"1. Reading video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # OpenCV BGR -> PIL RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
    cap.release()

    if not frames:
        raise ValueError("Video contains no frames or could not be read.")

    # --- USE IMPORTED UTILITY ---
    # This ensures sampling is identical to training
    indices = get_smart_indices_from_folder(frames)
    selected_frames = [frames[i] for i in indices]
    
    print(f"   Original frames: {len(frames)} | Selected frames: {len(selected_frames)}")

    # --- PREPROCESSING ---
    # TimmViTEncoder handles resizing/patching internally via its own transform
    # We just convert to Tensor here (0-1 range)
    transform = T.Compose([T.ToTensor()])
    
    tensor_list = [transform(img) for img in selected_frames]
    
    # Stack: [Time, C, H, W]
    video_tensor = torch.stack(tensor_list)
    return video_tensor

def main():
    args = get_args()
    device = torch.device(args.device)
    print(f"Running inference on {device}")

    # 1. Prepare Data
    video_tensor = load_video_tensor(args.video_path).to(device)

    # 2. Setup UNI2-h Encoder (Using your imported class)
    print("2. Initializing UNI2-h Encoder...")
    
    # --- EXACT KWARGS FROM YOUR TRAINING CODE ---
    timm_kwargs = {
        'pretrained': True,
        'img_size': 224, 
        'patch_size': 16, 
        'depth': 24,
        'num_heads': 24,
        'init_values': 1e-5, 
        'embed_dim': 1536,  # Output dimension
        'mlp_ratio': 2.66667*2,
        'num_classes': 0, 
        'no_embed_class': True,
        'mlp_layer': timm.layers.SwiGLUPacked, 
        'act_layer': torch.nn.SiLU, 
        'reg_tokens': 8, 
        'dynamic_img_size': True
    }
    
    try:
        # Initialize your imported TimmViTEncoder
        # Note: HF_TOKEN env var is needed inside here for huggingface access
        encoder = TimmViTEncoder(model_name='hf-hub:MahmoodLab/UNI2-h', kwargs=timm_kwargs)
        encoder.to(device)
        encoder.eval()
    except Exception as e:
        print(f"\n[Error] Initializing Encoder: {e}")
        print("Tip: Ensure HF_TOKEN is set.\n")
        sys.exit(1)

    # 3. Extract Features
    print("   Extracting features...")
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            # Your encoder returns [T, 1536] (averaged patches)
            features = encoder(video_tensor) 
            
            # Convert back to float32 for the classification model
            features = features.float()

    # Add batch dimension -> [1, T, 1536]
    input_features = features.unsqueeze(0)
    print(f"   Features shape: {input_features.shape}")

    # 4. Setup Classifier (VTransAdaptive)
    print("3. Loading VTransAdaptive...")
    model = VTransAdaptive(
        num_classes=args.num_classes,
        ratio=args.ratio,
        dropout=args.dropout,
        hidden_dim=args.hidden_size, # 1536
        n_masked_patch=args.n_masked_patch,
        mask_drop=args.mask_drop
    ).to(device)

    # Load Weights
    print(f"   Loading checkpoint: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Handle 'state_dict' key and 'module.' prefix from DataParallel training
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # 5. Predict
    print("4. Final Prediction...")
    with torch.no_grad():
        output = model(input_features)
        probs = torch.softmax(output, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()

    print("\n" + "="*40)
    print(f"VIDEO: {os.path.basename(args.video_path)}")
    print(f"PREDICTION: Class {pred_idx}")
    print(f"CONFIDENCE: {confidence:.2%}")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()