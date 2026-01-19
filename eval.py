from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import torch.nn as nn
from utils.datasets_utils import FeatureDataset, FeatureDataset_2, collate_fn
from models.vit_transformer_model import VTrans, VTransAdaptive
import torch
import wandb
from logging import WARNING, INFO
import logging
from utils.core_utils import test
import hydra
from omegaconf import OmegaConf, open_dict
import pandas as pd
import sys
from utils.core_utils import build_metrics



torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

wandb_token = os.getenv("WANDB_TOKEN")
wandb.login(wandb_token)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# --- wandb logging ---
run = wandb.init(
    project="video_classification",
    name="method"
)

data_paths = {
    'data' : 'datasets/feats',
    'csv' : 'datasets/csv'
}

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
   

   
logger = logging.getLogger('SmartfileTest')
logger.setLevel(INFO)
NEW_FORMAT = '[%(asctime)s] - [%(levelname)s] - %(message)s'
logger_format = logging.Formatter(NEW_FORMAT)
 
 
@hydra.main(config_path="configs", version_base=None)
def main(cfg: OmegaConf):
    logger.info("config: %s", cfg)
    seed_torch(int(cfg.seed))
    #------------Data--------------
    print("Init Dataset...\n")
    df_test = pd.read_csv(os.path.join(cfg.csv_path, 'val.csv'))
    #df_test = pd.read_csv(os.path.join(cfg.csv_path, 'val.csv'))
    
    test_slide_ids = df_test['slide_id'].tolist()
    test_labels = df_test['label_idx'].tolist()
    test_labels_raw = df_test['label'].tolist()
    print(test_labels[:10])
    
    
    test_data = FeatureDataset(feat_dir=cfg.feats_path,slide_ids=test_slide_ids, labels=test_labels, csv_dir = '/mnt/disk4/video-panninng-classification/datasets/csv/val.csv')
    #test_data = FeatureDataset_2(feat_dir=cfg.feats_path, slide_ids=test_slide_ids, labels=test_labels, labels_raw=test_labels_raw)
    feats, label, _ = test_data[0]
    
    print(f"Frames shape: {feats.shape}\nLabel: {label}")
    print("Done!")
    
    print("Init DataLoader...\n")
    test_loader = DataLoader(test_data, batch_size = cfg.batch_size, shuffle=False, num_workers=4,pin_memory=True,persistent_workers=True, collate_fn=collate_fn)
    print("Done!")
    
    #------------Model-------------
    main_model = VTransAdaptive(num_classes=int(cfg.num_classes),
                        ratio=float(cfg.ratio),
                        dropout= float(cfg.dropout), 
                        hidden_dim=int(cfg.hidden_size), n_masked_patch=cfg.n_masked_patch, mask_drop=cfg.mask_drop).to(device)
    main_model.load_state_dict(torch.load("results/new/baseline_ratio0.1_seed512.pth"))
    metrics = build_metrics(cfg.num_classes, device)
    try:
        test_loss_fn = nn.CrossEntropyLoss()
        test(run, logger, cfg, main_model, test_loader, device, metrics, stride=5)
    except Exception as e:
        logger.error(f"Error occurred during training: {e}")
        torch.save(main_model.state_dict(), os.path.join(f"results/error_model.pth"))
        wandb.finish()
        sys.exit(1)

if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main() 