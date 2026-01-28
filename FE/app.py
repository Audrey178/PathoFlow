import os
import sys
import torch
import torch.nn.utils.prune as prune
import torch.utils.data.dataloader
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

_OriginalDataLoader = torch.utils.data.dataloader.DataLoader

class SafeDataLoader(_OriginalDataLoader):
    def __init__(self, *args, **kwargs):
        # Force single-process loading
        kwargs['num_workers'] = 0
        kwargs['persistent_workers'] = False
        super().__init__(*args, **kwargs)

# Overwrite the class globally
torch.utils.data.DataLoader = SafeDataLoader
torch.utils.data.dataloader.DataLoader = SafeDataLoader
# -----------------------------------

import cv2
import timm
import numpy as np
import gc
import tempfile
import traceback
from PIL import Image
from flask import Flask, request, jsonify, render_template
from torchvision import transforms as T
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
UTILS_DIR = os.path.join(PARENT_DIR, 'utils')
sys.path.append(PARENT_DIR)
sys.path.append(UTILS_DIR)

from utils.feat_extract import TimmViTEncoder
from models.vit_transformer_model import VTransAdaptive
from utils.find_frames_idx import get_smart_indices_from_frames
# --- 3. CONFIGURATION ---
app = Flask(__name__, template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 5000 * 1024 * 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = os.environ.get("MODEL_PATH", "path_to_default_model.pth")
CLASS_NAMES = {0: "Normal", 1: "Adenoma", 2: "Malignant"}
MODELS = {}

# --- 4. MODEL LOADING ---
def load_models_if_needed():
    if "encoder" in MODELS and "classifier" in MODELS:
        return

    print(f"[{os.getpid()}] Loading models on {DEVICE}...", flush=True)
    
    timm_kwargs = {
        'pretrained': True, 'img_size': 224, 'patch_size': 16, 'depth': 24,
        'num_heads': 24, 'init_values': 1e-5, 'embed_dim': 1536,
        'mlp_ratio': 2.66667*2, 'num_classes': 0, 'no_embed_class': True,
        'mlp_layer': timm.layers.SwiGLUPacked, 'act_layer': torch.nn.SiLU, 
        'reg_tokens': 8, 'dynamic_img_size': True
    }

    try:
        encoder = TimmViTEncoder(model_name='hf-hub:MahmoodLab/UNI2-h', kwargs=timm_kwargs)
        encoder.to(DEVICE)
        encoder.eval()
        MODELS["encoder"] = encoder
        
        classifier = VTransAdaptive(
            num_classes=len(CLASS_NAMES), ratio=0.1, dropout=0.5, 
            hidden_dim=1536, n_masked_patch=10, mask_drop=0.1
        )
        if os.path.exists(MODEL_PATH):
            print(f"Loading checkpoint: {MODEL_PATH}", flush=True)
            checkpoint = torch.load(MODEL_PATH, map_location='cpu')
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            classifier.load_state_dict(state_dict, strict=False)
            classifier.to(DEVICE)
            for name, module in classifier.named_modules():
                if isinstance(module, torch.nn.Linear):
                    prune.l1_unstructured(module, name='weight', amount=0.2)
                    prune.remove(module, 'weight')
            
            classifier.eval()
            MODELS["classifier"] = classifier
        else:
            print(f"WARNING: Checkpoint NOT found at {MODEL_PATH}", flush=True)
            raise FileNotFoundError(f"Checkpoint missing: {MODEL_PATH}")
        print("Models loaded successfully.", flush=True)
    except Exception as e:
        print(f"Model Load Error: {e}", flush=True)
        raise RuntimeError(f"Encoder Load Failed: {e}")
    
def preprocess_video(video_path):
    logging.info(f"DEBUG: Starting extraction for {video_path}")
    cap = cv2.VideoCapture(video_path)
    all_frames = []
    
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        all_frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        count += 1
        if count % 100 == 0:
            logging.debug(f"DEBUG: Extracted {count} frames")

    cap.release()
    if os.path.exists(video_path):
        os.remove(video_path)
    logging.info(f"DEBUG: Completed extraction. Total: {len(all_frames)}")

    indices = get_smart_indices_from_frames(all_frames)
    logging.info(f"DEBUG: Smart indices selected: {indices}")
    
    selected_images = [all_frames[i] for i in indices]
    
    del all_frames
    gc.collect()
    logging.debug("DEBUG: Large frame list deleted and GC called.")
    
    return torch.stack([T.ToTensor()(img) for img in selected_images]).to(DEVICE)

# --- 6. ROUTES ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'running', 'device': DEVICE}), 200
       
@app.route('/predict', methods=['POST'])
def predict():
    temp_path = None
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No file'}), 400
        load_models_if_needed()
        f = request.files['video']
        fd, temp_path = tempfile.mkstemp(suffix=".mp4", dir="/app") 
        os.close(fd)
        f.save(temp_path)
        
        # 1. Process Video -> List -> Smart Indices -> Tensor
        input_tensor = preprocess_video(temp_path)
        
        # 2. Chunked Inference for Encoder
        chunk_size = 25
        all_feats = []
        
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                for i in range(0, input_tensor.size(0), chunk_size):
                    chunk = input_tensor[i : i + chunk_size]
                    chunk_feat = MODELS["encoder"](chunk).float()
                    all_feats.append(chunk_feat)
                    torch.cuda.empty_cache()
                feats = torch.cat(all_feats, dim=0)
                del input_tensor, all_feats
                gc.collect()
                torch.cuda.empty_cache()

            # 3. Classifier Inference
            feats = feats.unsqueeze(0) 
            attn_mask = torch.zeros((1, feats.size(1)), dtype=torch.bool).to(DEVICE)
            
            out = MODELS["classifier"](feats, attn_mask, epoch=1, stride=1)
            probs = torch.softmax(out, dim=1)
            idx = torch.argmax(probs, dim=1).item()
            del feats, attn_mask
            torch.cuda.empty_cache()
            
        return jsonify({
            'predicted_class': CLASS_NAMES.get(idx, str(idx)),
            'confidence': f"{probs[0][idx].item():.2%}",
            'probabilities': {CLASS_NAMES.get(i, i): f"{p:.2%}" for i, p in enumerate(probs[0].tolist())}
        })
            
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8502, debug=False, use_reloader=False)