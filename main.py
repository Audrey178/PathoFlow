from dotenv import load_dotenv
import numpy as np
import os
import hydra
from torch.utils.data import DataLoader
import torch
from omegaconf import OmegaConf
from logging import INFO
import logging
from utils.datasets_utils import FeatureDataset, collate_fn
from models.vit_transformer_model import VTrans, VTrans_Bertmasking, VTransDuo, VTransAdaptive, VTransUnified
from utils.core_utils import train
import wandb
import sys
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# bật SDPA backend (FlashAttention, mem-efficient, math)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

load_dotenv()


wandb_token = os.getenv("WANDB_TOKEN")
wandb.login(wandb_token)


# --- Setup logging ---
logger = logging.getLogger('SmartfileTest')
logger.setLevel(INFO)
NEW_FORMAT = '[%(asctime)s] - [%(levelname)s] - %(message)s'
logger_format = logging.Formatter(NEW_FORMAT)
# # --- wandb logging ---
# run = wandb.init(
#     project="video_classification",
#     name="method"
# )


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
    
@hydra.main(config_path="configs", version_base=None)
def main(cfg: OmegaConf):
    logger.info("config: %s", cfg)
    seed_torch(int(cfg.seed))
    experiment_name = getattr(cfg, "save_name", "manual_run")
    
    run = wandb.init(
        project="video_classification",
        name = experiment_name,
        config= OmegaConf.to_container(cfg, resolve=True))

    #------------Data--------------
    print("Init Dataset...\n")
    df_train = pd.read_csv(os.path.join(cfg.csv_path, 'train.csv'))
    df_val = pd.read_csv(os.path.join(cfg.csv_path, 'test.csv'))
    
    train_slide_ids = df_train['slide_id'].tolist()
    train_labels = df_train['label_idx'].tolist()
    train_labels_raw = df_train['label'].tolist()
    print(train_labels[:10])
    
    val_slide_ids = df_val['slide_id'].tolist()
    val_labels = df_val['label_idx'].tolist()
    val_labels_raw = df_val['label'].tolist()
    
    train_data = FeatureDataset(feat_dir=cfg.feats_path,slide_ids=train_slide_ids, labels=train_labels, csv_dir = '/mnt/disk4/video-panninng-classification/datasets/csv/train.csv')
    val_data = FeatureDataset(feat_dir=cfg.feats_path,slide_ids=val_slide_ids, labels=val_labels, csv_dir = '/mnt/disk4/video-panninng-classification/datasets/csv/val.csv')

    feats, label, _ = train_data[0]
    
    print(f"Frames shape: {feats.shape}\nLabel: {label}")
    print("Done!")
    
    print("Init DataLoader...\n")
    train_loader = DataLoader(train_data, batch_size = cfg.batch_size, shuffle=True, num_workers=4,persistent_workers=True, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data , batch_size = cfg.batch_size, shuffle=False, num_workers=4,persistent_workers=True,pin_memory=True, collate_fn=collate_fn)
    print("Done!")
    #------------Model-------------
    # main_model = VTransDuo(num_classes=int(cfg.num_classes),
    #                     dropout= float(cfg.dropout), 
    #                     hidden_dim=int(cfg.hidden_size), n_masked_patch=cfg.n_masked_patch, mask_drop=cfg.mask_drop).to(device)
    main_model = VTransAdaptive(num_classes=int(cfg.num_classes),
                        ratio=float(cfg.ratio),
                        dropout= float(cfg.dropout), 
                        hidden_dim=int(cfg.hidden_size), n_masked_patch=cfg.n_masked_patch, mask_drop=cfg.mask_drop).to(device)
    #
    # main_model = VTransUnified(num_classes=int(cfg.num_classes),
    #                 ratio=float(cfg.ratio),
    #                 dropout= float(cfg.dropout), 
    #                 hidden_dim=int(cfg.hidden_size), n_masked_patch=cfg.n_masked_patch, mask_drop=cfg.mask_drop, dual_branch=True).to(device)
  
    #-------------Train------------
    try:
        train(run, logger, cfg, train_loader, val_loader, device, main_model)
        name_prefix = getattr(cfg, "save_name", "best_model")
        filename = 'final/' + f"{name_prefix}.pth"
        save_path = os.path.join(cfg.results_dir_path, filename)
        torch.save(main_model.state_dict(), save_path)
        
    except Exception as e:
        logger.error(f"Error occurred during training: {e}")
        torch.save(main_model.state_dict(), os.path.join(f"results/error_model.pth"))
        wandb.finish()
        sys.exit(1)

if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main() 
   
    
    
    