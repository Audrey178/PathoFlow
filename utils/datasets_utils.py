from torch.utils.data import Dataset
import torch
import os
from pathlib import Path
import csv
import json
         
class FrameDataset(Dataset):
    def __init__(self, slide_ids, labels,  root_dir = 'datasets/frames_v3', img_size = 224, transform = None):
        super().__init__()
        self.slide_ids = slide_ids
        self.labels = labels
        self.folder_dirs = [Path(root_dir) / label / slide_id for (slide_id, label) in zip(slide_ids, labels)]
        self.img_size = img_size

    def __len__(self):
        return len(self.folder_dirs)

    def __getitem__(self, index):
        folder = self.folder_dirs[index]
        slide_id = self.slide_ids[index]
        img_files = sorted(os.listdir(folder))
        img_paths = [os.path.join(folder, f) for f in img_files]
        return img_paths, slide_id
    
    
def find_entry_csv(csv_path, target_slide_id):
    target_slide_id = str(target_slide_id)
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        # Iterate through rows one by one
        for row in reader:
            # Check if the column exists and matches
            if row.get('slide_id') == target_slide_id:
                return row
 
class FeatureDataset(Dataset):
    def __init__(self, feat_dir, slide_ids, labels, csv_dir):
        super().__init__()
        self.feat_dir = feat_dir
        self.slide_ids = slide_ids
        self.labels = labels
        self.csv_dir = csv_dir

    def __len__(self):
        return len(self.slide_ids)

    def __getitem__(self, index):
        slide_id = self.slide_ids[index]
        label = self.labels[index]
        feat_path = os.path.join(self.feat_dir, f'{slide_id}.pt')
        
        # USE OPTICAL FLOW UNCOMMENT THIS
        # origin_frames_path = find_entry_csv(self.csv_dir, slide_id)['path'][:-4].replace('videos', 'frames_v3')
        # indices = find_entry_csv(self.csv_dir, slide_id)['selected_frames_new_new_new_new']
        # indices_list = json.loads(indices)
        # indices_tensor = torch.tensor(indices_list, dtype=torch.long)
        features_raw = torch.load(feat_path)  # [T, D]
        # features = features_raw[indices_tensor]
        
        length = features_raw.shape[0]
        # length = features.shape[0]
        
        return features_raw, label, length 
        # return features, torch.tensor(label) 

class FeatureDataset_2(Dataset): # Nam
    def __init__(self, feat_dir, slide_ids, labels, labels_raw):
        super().__init__()
        self.feat_dir = feat_dir
        self.slide_ids = slide_ids
        self.labels = labels
        self.labels_raw = labels_raw

    def __len__(self):
        return len(self.slide_ids)

    def __getitem__(self, index):
        slide_id = self.slide_ids[index]
        label = self.labels[index]
        label_raw = self.labels_raw[index]
        feat_path = os.path.join(self.feat_dir, f'{label_raw}/{slide_id}.pt')
        
        features_raw = torch.load(feat_path)
        
        length = features_raw.shape[0]
        
        return features_raw, label, length     
        
def collate_fn(batch):
    feats, labels, lengths = zip(*batch) 

    # Pad thành [B, T_max, 3, H, W]
    feats = torch.nn.utils.rnn.pad_sequence(
        feats, batch_first=True
    )
    labels = torch.tensor(labels, dtype=torch.long)
    lengths = torch.tensor(lengths, dtype=torch.long)

    B, T , _ = feats.shape
    attn_mask = torch.arange(T).unsqueeze(0) >= lengths.unsqueeze(1)  # [B, T]
    
    return feats, labels, attn_mask

   

        
