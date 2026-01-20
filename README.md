# PathoFlow: A Motion-Aware Prototype-Based Framework for Microscopic Pathology Video Classification

## Overview

This repository contains the implementation of PathoFlow, a framework for video analysis with adaptive token selection and masking strategies. The codebase includes data preparation, feature extraction, model training, and evaluation pipelines.

We provide a subset of videos to facilitate reproducibility of our results: [Download](https://mega.nz/folder/1jBmnCza#eAfKKdBksybicLvA2kSsqA). For your own dataset, follow the steps below to prepare data.

## Table of Contents

1. [Installation](#installation)
2. [Data Preparation](#data-preparation)
3. [Training](#training)
4. [Evaluation](#evaluation)
5. [Project Structure](#project-structure)

## Installation

### Requirements

- Python 3.8+
- CUDA 11.0+ (for GPU support)
- FFmpeg (for video processing)

### Setup

1. Clone the repository and navigate to the project directory:

```bash
cd PathoFlow
```

2. Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up environment variables (for HuggingFace token, WanDB token):

Create a `.env` file in the project root:

```
HF_TOKEN=your_huggingface_token_here
```

## Data Preparation

### Overview

For your own dataset, you can modify and run following this steps: 

### Step 1: Extract Frames from Videos

Convert your videos to frame sequences:

```bash
python preprocessing_video.py
```

### Step 2: Extract Features Using SSL Pretrained Models

We recommend extracting features using self-supervised learning (SSL) pretrained models. Our implementation uses the **UNI2-h** checkpoint provided by MahmoodLab.

Extract features for each dataset split:

```bash
# For training split
python utils/feat_extract.py \
    --data_dir datasets \
    --split train \
    --feats_dir datasets/feats

# For validation split
python utils/feat_extract.py \
    --data_dir datasets \
    --split val \
    --feats_dir datasets/feats

# For test split
python utils/feat_extract.py \
    --data_dir datasets \
    --split test \
    --feats_dir datasets/feats
```

### Step 3: Prepare Dataset CSV Files

Create CSV files for train/val/test splits in `datasets/csv/`:

**Format:** `{split}.csv`

```csv
slide_id,label,label_idx,path
video_001,Normal,0,datasets/videos/video_001.mp4
video_002,Adenoma,1,datasets/videos/video_002.mp4
...
```

### Step 4: Optical Flow-based Frame Selection

To reduce redundancy and select keyframes based on motion patterns, run the optical flow frame selection scripts for each dataset split:

```bash
# For training split
python utils/finding_frame_idx_train.py

# For validation split
python utils/finding_frame_idx_val.py

# For test split
python utils/finding_frame_idx_test.py
```

These scripts analyze video frames using **optical flow** to intelligently select representative frames based on:
- **Motion detection**: Tracks pixel-level motion between consecutive frames
- **Motion patterns**: Classifies scanning modes (SLOW_SWEEP, NORMAL_RASTER, FAST_SWEEP)
- **Adaptive sampling**: Selects frames based on accumulated motion distance
- **Turn detection**: Detects direction changes in raster scanning patterns

The algorithm detects:
- **MIN_FEATURES**: Minimum texture features required (50)
- **MIN_MOTION_PX**: Minimum motion threshold in pixels (2)
- **MAX_JITTER_PX**: Maximum jitter tolerance (3)
- **OVERLAP_RATIO**: Overlap ratio for stitching (0.2)

The updated CSV will include a `selected_frames` column containing the indices of keyframes:

```csv
slide_id,label,label_idx,path,selected_frames
video_001,Normal,0,datasets/videos/video_001.mp4,"[0, 5, 12, 28, 45, ...]"
video_002,Adenoma,1,datasets/videos/video_002.mp4,"[0, 8, 19, 31, 52, ...]"
...
```


## Training

### Baseline Model

To train the baseline:

```bash
bash run.sh
```

Or manually:

```bash
python main.py --config-name baseline
```

### Configuration

Edit `configs/baseline.yaml` to modify training parameters:

```yaml
strategy: 'baseline'
seed: 512
batch_size: 8
num_classes: 3
max_epochs: 100
lr: 2e-4
dropout: 0.2
top_k: 256          # Number of tokens to keep
n_masked_patch: 0   # Number of patches to mask
mask_drop: 0.5      # Mask dropout probability
```

### Hyperparameter Ablation Studies

#### Top-K Ablation

Test different numbers of tokens to keep:

```bash
bash run_topk_fixed.sh
```

Tests top_k values: [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

#### Masking Strategy

Evaluate different masking configurations:

```bash
bash run_masking.sh
```

Tests combinations of:
- `n_masked_patch`: [4, 8, 12]
- `mask_drop`: [0.5]

#### Adaptive Token Selection

Run experiments with multiple seeds:

```bash
bash run_adaptive.sh
```

Seeds: [0.05, 0.07, 0.1, 0.25, 0.5, 0.75]

### Training with Custom Parameters

Override config parameters from command line:

```bash
python main.py \
    --config-name baseline \
    seed=1024 \
    batch_size=16 \
    lr=1e-4 \
    top_k=512 \
    save_name="custom_experiment"
```

## Evaluation

Evaluate trained models:

```bash
python eval.py --model_path results/baseline/best_model.pt
```

Or using the provided evaluation script:

```bash
bash eval.sh
```

## Inference on Single Video

Run inference on a single video file to get predictions:

```bash
python infer.py \
    --video_path path/to/your/video.mp4 \
    --model_path results/baseline/best_model.pth
```

### Arguments

**Required:**
- `--video_path`: Path to input video file (.mp4, .avi, .mkv)
- `--model_path`: Path to trained model checkpoint (.pth)

**Optional:**
- `--num_classes`: Number of output classes (default: 2)
- `--hidden_size`: Embedding dimension (default: 1536, should match UNI2-h)
- `--ratio`: Token selection ratio (default: 0.1)
- `--dropout`: Dropout probability (default: 0.5)
- `--n_masked_patch`: Number of patches to mask (default: 10)
- `--mask_drop`: Mask dropout probability (default: 0.1)
- `--device`: Device to use (default: 'cuda' if available, else 'cpu')

### Inference Pipeline

The inference script performs the following steps:

1. **Load Video**: Reads all frames from the input video file
2. **Frame Selection**: Applies optical flow-based smart sampling to select keyframes
3. **Feature Extraction**: Extracts features using UNI2-h pretrained encoder
4. **Classification**: Passes features through trained VTransAdaptive model
5. **Output**: Returns prediction class and confidence score

### Example Usage

```bash
# Basic inference
python infer.py \
    --video_path datasets/videos/sample.mp4 \
    --model_path results/baseline/checkpoint_best.pth

# With custom parameters
python infer.py \
    --video_path datasets/videos/sample.mp4 \
    --model_path results/baseline/checkpoint_best.pth \
    --num_classes 3 \
    --ratio 0.15 \
    --n_masked_patch 12 \
    --device cuda
```

### Output Example

```
1. Reading video: datasets/videos/sample.mp4
   Original frames: 1250 | Selected frames: 125
2. Initializing UNI2-h Encoder...
   Extracting features...
   Features shape: torch.Size([1, 125, 1536])
3. Loading VTransAdaptive...
   Loading checkpoint: results/baseline/checkpoint_best.pth
4. Final Prediction...

========================================
VIDEO: sample.mp4
PREDICTION: Class 1
CONFIDENCE: 95.32%
========================================
```

### Notes

- Ensure `HF_TOKEN` is set in your `.env` file for HuggingFace model access
- The inference uses the same optical flow-based frame selection as training


## Key Features

- **Adaptive Token Selection**: Dynamically select relevant tokens based on optical flow analysis
- **Masking Strategies**: Evaluate different masking approaches for robustness
- **Multi-GPU Support**: Efficient training with distributed data parallel
- **Weights & Biases Integration**: Experiment tracking via W&B

## Performance Monitoring

Training metrics are logged to Weights & Biases (W&B). Configure in `baseline.yaml`:

```yaml
wandb: True  # Set to False to disable W&B logging
```

View experiments at: https://wandb.ai/

## Video demo

https://github.com/Audrey178/PathoFlow/releases/download/v1.0-demo/clean_demo.mp4
```bash
clean_demo.mp4
```


## Citation
If you use this code in your research, please cite:

```bibtex
@article{PathoFlow2026,
  title={PathoFlow:  A Motion-Aware Prototype-Based Framework for Microscopic Pathology Video Classification},
  author={Your Name},
  journal={Your Journal},
  year={2026}
}
```

## License

[Specify your license here]

## Contact

For questions or issues, please contact: [your-email@example.com]
