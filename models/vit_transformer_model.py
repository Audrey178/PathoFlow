import torch 
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class ClusterTokensPooling(nn.Module):
    def __init__(self, num_clusters, dim, heads=4):
        super().__init__()
        self.num_clusters = num_clusters
        self.cluster_tokens = nn.Parameter(torch.randn(1, num_clusters, dim))
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, attn_mask=None):
        # x: [B, N, D] (patch/frame embeddings)
        B, T, D = x.shape 
        cluster_tokens = self.cluster_tokens.expand(B, -1, -1)  # [B, num_clusters, D]
        
        if attn_mask is not None:
        # Check if any row is entirely True (meaning everything is masked)
            is_all_masked = attn_mask.all(dim=1)  # [B]
            
            if is_all_masked.any():
                # If a row is all masked, we must unmask at least one value to prevent NaN.
                # We unmask the first token (index 0) so Softmax has something to calculate.
                # (We clone to avoid modifying the mask for other layers if shared)
                attn_mask = attn_mask.clone()
                attn_mask[is_all_masked, 0] = False
        
        # cluster tokens làm query, còn patch/frame làm key/value
        out, _ = self.attn(query=cluster_tokens, key=x, value=x, key_padding_mask=attn_mask)  # [B, num_clusters, D]
        
        out = self.norm(out)
        return out

class AdaptiveCrossAttentionPooling(nn.Module):
    def __init__(self, dim, ratio=0.25, max_tokens=8192, heads=8):
        super().__init__()
        self.dim = dim
        self.ratio = ratio
        self.max_tokens = max_tokens
        self.heads = heads
        
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        
        self.query_gen = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        
    def forward(self, x, attn_mask=None):
        B, T, D = x.shape
        
        if attn_mask is not None:
            valid_len = attn_mask.sum(dim=1).float().mean()
            k = max(4, int(valid_len * self.ratio))
        else:
            k = max(4, int(T * self.ratio))
        
        n_tokens = min(k, self.max_tokens)
        
        queries = F.adaptive_avg_pool1d(x.transpose(1, 2), n_tokens).transpose(1, 2)  # [B, n_tokens, D]
        
        out, _ = self.cross_attn(query=queries, key=x, value=x, key_padding_mask=attn_mask)  # [B, n_tokens, D]
        
        out = self.norm(out)
        return out


# class VTrans(nn.Module):
#     def __init__(self, num_classes, num_clusters = 500, chunk_size = 500, dropout = 0.15, hidden_dim = 1536, model = None):
#         super().__init__()
#         # self.vit_pretrained = timm.create_model(model_name=model_name, pretrained=True, **timm_kwargs)
#         self.vit_pretrained = model
#         self.vit_pretrained.set_grad_checkpointing(True)
        
#         self.hidden_dim = hidden_dim
#         self.num_classes = num_classes
#         self.num_clusters = num_clusters
#         self.chunk_size = chunk_size
        
#         self.cluster_pool = ClusterTokensPooling(num_clusters=num_clusters, dim=hidden_dim, heads=8)
        
#         self.dropout = dropout
#         self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
#         encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True, dropout=dropout)
#         self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
#         self.fc = nn.Linear(hidden_dim, num_classes)
    
#     def forward(self, frames):  # [T, 3, 224, 224]
#         B, T, C, H, W = frames.shape
#         frames = frames.view(-1, C, H, W)  # [B*T, 3, H, W]
#         chunk_size = min(self.chunk_size, T)
        
#         all_embeds = []
#         for i in range(0, B*T, chunk_size):
#             chunk = frames[i:i+chunk_size]
#             with torch.cuda.amp.autocast(dtype=torch.float16):
#                 embeds = self.vit_pretrained(chunk)  # [chunk_size, hidden_dim]
#             all_embeds.append(embeds)
#         embeds = torch.cat(all_embeds, dim=0)
        
#         # reshape lại thành [B, T, hidden_dim]
#         embeds = embeds.view(B, T, -1) #[B, T, hidden_dim]
        
#         # ----> Cluster pooling (learnable)
#         clustered_feats = self.cluster_pool(embeds)  # [B, num_clusters, hidden_dim]

#         cls = self.cls_token.expand(B, -1, -1)
#         out = torch.cat([cls, clustered_feats], dim=1)  
#         # ----> Temporal encoder
#         out = self.temporal_encoder(out)  # [B, num_clusters, hidden_dim]

#         # ----> Pool cluster tokens (mean hoặc lấy [CLS])
#         out = out[:, 0]  # [B, hidden_dim]

#         # ----> Classification
#         logits = self.fc(out)  # [B, num_classes]
#         return logits
    
class VTrans(nn.Module):
    def __init__(self, num_classes, num_clusters = 256, chunk_size = 500, dropout = 0.15, hidden_dim = 1536, n_masked_patch=5, mask_drop=0.5):
        super().__init__()
        # self.vit_pretrained = timm.create_model(model_name=model_name, pretrained=True, **timm_kwargs)
        # self.vit_pretrained = model
        # self.vit_pretrained.set_grad_checkpointing(True)
        self.n_masked_patch = n_masked_patch 
        self.mask_drop = mask_drop 
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_clusters = num_clusters
        self.chunk_size = chunk_size
        self.importance_gate = nn.Linear(hidden_dim, 1)
        self.cluster_pool = ClusterTokensPooling(num_clusters=num_clusters, dim=hidden_dim, heads=8)
        
        self.dropout = dropout
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True, dropout=dropout)
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, feats, attn_mask, epoch, stride = 1):  # [T, hidden_dim]
        B, T, D = feats.shape
        
        idxs = (torch.arange(0, T, stride, device=feats.device) + (epoch % stride)) % T
       
        feats = feats[:, idxs]  # [B, T//stride, D]
        attn_mask = attn_mask[:, idxs]  # [B, T//stride]    
        if self.n_masked_patch > 0 and self.training: #Huy
            # 1. Get importance scores
            importance = self.importance_gate(feats).squeeze(-1) 
            
            low_val = -torch.finfo(importance.dtype).max / 2
            importance = importance.masked_fill(attn_mask == 0, low_val)
            
            # 3. Find top-k
            n_mask = min(self.n_masked_patch, T)
            _, indices = torch.topk(importance, n_mask, dim=-1)
            
            # 4. Handle the dropping
            num_to_drop = int(n_mask * self.mask_drop)
            if num_to_drop > 0:
                # Create a random mask on the same device and same dtype
                # Shuffling indices to pick which to drop
                noise = torch.rand_like(indices.float())
                rand_select = torch.argsort(noise, dim=-1)[:, :num_to_drop]
                to_mask_indices = indices.gather(1, rand_select)

                # 5. CREATE MASK WITH EXPLICIT DTYPE
                # If this isn't feats.dtype, the multiplication below triggers the overflow
                m_mask = torch.ones_like(attn_mask, dtype=feats.dtype) 
                
                # Use scatter_ with a value of 0.0 (safe for all dtypes)
                m_mask.scatter_(1, to_mask_indices, 0.0)
                
                # Update the attention mask
                attn_mask = attn_mask * m_mask
        # ----> Cluster pooling (learnable)
        clustered_feats = self.cluster_pool(feats, attn_mask)  # [B, num_clusters, hidden_dim]

        cls = self.cls_token.expand(B, -1, -1)
        out = torch.cat([cls, clustered_feats], dim=1)  
        
        # ----> Temporal encoder
        out = self.temporal_encoder(out)  # [B, num_clusters, hidden_dim]

        # ----> Pool cluster tokens (mean hoặc lấy [CLS])
        cls = out[:, 0]  # [B, hidden_dim]
        mean = out[:, 1:].mean(dim=1)
        out = cls*0.8 + mean*0.2  # [B, hidden_dim]

        # ----> Classification
        logits = self.fc(out)  # [B, num_classes]
        return logits
    
class VTransAdaptive(nn.Module):
    def __init__(self, num_classes, chunk_size = 500, dropout = 0.15, hidden_dim = 1536, n_masked_patch=5, mask_drop=0.5, ratio=0.5):
        super().__init__()
        self.n_masked_patch = n_masked_patch 
        self.mask_drop = mask_drop 
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.chunk_size = chunk_size
        self.importance_gate = nn.Linear(hidden_dim, 1)
        self.token_pooler = AdaptiveCrossAttentionPooling(dim=hidden_dim, ratio=ratio, heads=8)
        
        self.dropout = dropout
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True, dropout=dropout)
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, feats, attn_mask, epoch, stride = 1):  # [T, hidden_dim]
        B, T, D = feats.shape
        
        idxs = (torch.arange(0, T, stride, device=feats.device) + (epoch % stride)) % T
       
        feats = feats[:, idxs]  # [B, T//stride, D]
        attn_mask = attn_mask[:, idxs]  # [B, T//stride]    
        if self.n_masked_patch > 0 and self.training: #Huy
            # 1. Get importance scores
            importance = self.importance_gate(feats).squeeze(-1) 
            
            low_val = -torch.finfo(importance.dtype).max / 2
            importance = importance.masked_fill(attn_mask == 0, low_val)

            # 3. Find top-k
            n_mask = min(self.n_masked_patch, T)
            _, indices = torch.topk(importance, n_mask, dim=-1)

            # 4. Handle the dropping
            num_to_drop = int(n_mask * self.mask_drop)
            if num_to_drop > 0:
                # Create a random mask on the same device and same dtype
                # Shuffling indices to pick which to drop
                noise = torch.rand_like(indices.float())
                rand_select = torch.argsort(noise, dim=-1)[:, :num_to_drop]
                to_mask_indices = indices.gather(1, rand_select)

                # 5. CREATE MASK WITH EXPLICIT DTYPE
                # If this isn't feats.dtype, the multiplication below triggers the overflow
                m_mask = torch.ones_like(attn_mask, dtype=feats.dtype) 
                
                # Use scatter_ with a value of 0.0 (safe for all dtypes)
                m_mask.scatter_(1, to_mask_indices, 0.0)
                
                # Update the attention mask
                attn_mask = attn_mask * m_mask
                
        # ----> Cluster pooling (learnable)
        clustered_feats = self.token_pooler(feats, attn_mask)  # [B, num_clusters, hidden_dim]

        cls = self.cls_token.expand(B, -1, -1)
        out = torch.cat([cls, clustered_feats], dim=1)  
        
        # ----> Temporal encoder
        out = self.temporal_encoder(out)  # [B, num_clusters, hidden_dim]

        # ----> Pool cluster tokens (mean hoặc lấy [CLS])
        cls = out[:, 0]  # [B, hidden_dim]
        mean = out[:, 1:].mean(dim=1)
        out = cls*0.8 + mean*0.2  # [B, hidden_dim]

        # ----> Classification
        logits = self.fc(out)  # [B, num_classes]
        return logits

import torch
import torch.nn as nn

class VTransDuo(nn.Module):
    def __init__(self, num_classes, num_clusters=256, chunk_size=500, dropout=0.15, 
                 hidden_dim=1536, n_masked_patch=5, mask_drop=0.5):
        super().__init__()
        # self.vit_pretrained = timm.create_model(...) 
        
        self.n_masked_patch = n_masked_patch 
        self.mask_drop = mask_drop 
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_clusters = num_clusters
        self.chunk_size = chunk_size
        
        # Attention / Importance Gate
        self.importance_gate = nn.Linear(hidden_dim, 1)
        
        # Pooling & Encoders
        # Assuming ClusterTokensPooling is defined elsewhere in your code
        self.cluster_pool = ClusterTokensPooling(num_clusters=num_clusters, dim=hidden_dim, heads=8)
        
        self.dropout = dropout
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True, dropout=dropout)
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def _backend_forward(self, feats, mask):
        """
        Shared backend that processes features after the masking stage.
        Ensures both clean and masked branches use the exact same weights.
        """
        B = feats.shape[0]
        
        # ----> Cluster pooling (learnable)
        clustered_feats = self.cluster_pool(feats, mask)  # [B, num_clusters, hidden_dim]

        cls = self.cls_token.expand(B, -1, -1)
        out = torch.cat([cls, clustered_feats], dim=1)  
        
        # ----> Temporal encoder
        out = self.temporal_encoder(out)  # [B, num_clusters + 1, hidden_dim]

        # ----> Pool cluster tokens
        cls_out = out[:, 0]  # [B, hidden_dim]
        mean_out = out[:, 1:].mean(dim=1)
        
        # Your custom fusion logic
        final_embed = cls_out * 0.8 + mean_out * 0.2  # [B, hidden_dim]

        # ----> Classification
        logits = self.fc(final_embed)  # [B, num_classes]
        
        return logits

    def forward(self, feats, attn_mask, epoch, stride=1):  # [B, T, hidden_dim]
        B, T, D = feats.shape
        
        # 1. Strided Sampling / Slicing
        idxs = (torch.arange(0, T, stride, device=feats.device) + (epoch % stride)) % T
        feats = feats[:, idxs]       # [B, T//stride, D]
        attn_mask = attn_mask[:, idxs] # [B, T//stride]
        
        # ---------------------------------------------------------
        # BRANCH 1: CLEAN PATH (Always run this)
        # This anchors the gradients so the model doesn't drift
        # ---------------------------------------------------------
        logits_clean = self._backend_forward(feats, attn_mask)

        # ---------------------------------------------------------
        # BRANCH 2: MASKED PATH (Training only)
        # This forces the model to look for non-obvious features
        # ---------------------------------------------------------
        if self.training and self.n_masked_patch > 0:
            # -- Importance Calculation --
            # Detach feats here if you want to prevent the gate from "hacking" the loss,
            # but standard MIL usually keeps gradients flowing.
            importance = self.importance_gate(feats).squeeze(-1) 
            low_val = -torch.finfo(importance.dtype).max / 2
            importance = importance.masked_fill(attn_mask == 0, low_val)

            # -- Top-k Selection --
            # Recalculate T because of stride
            current_T = feats.shape[1]
            n_mask = min(self.n_masked_patch, current_T)
            _, indices = torch.topk(importance, n_mask, dim=-1)

            logits_masked_list = []
            for _ in range(4):
                # -- Stochastic Dropping --
                num_to_drop = int(n_mask * self.mask_drop)
                
                if num_to_drop > 0:
                    noise = torch.rand_like(indices.float())
                    rand_select = torch.argsort(noise, dim=-1)[:, :num_to_drop]
                    to_mask_indices = indices.gather(1, rand_select)

                    # -- Mask Creation --
                    m_mask = torch.ones_like(attn_mask, dtype=feats.dtype) 
                    m_mask.scatter_(1, to_mask_indices, 0.0)
                    
                    # Combine with original mask
                    masked_attn_mask = attn_mask * m_mask
                    
                    # Pass through the SHARED backend
                    logits_masked = self._backend_forward(feats, masked_attn_mask)
                    logits_masked_list.append(logits_masked)
                    # Return BOTH logits
            if len(logits_masked_list) > 0:
                return logits_clean, logits_masked_list

            # If evaluating, or if masking is off, just return clean logits
        return logits_clean

class VTrans_Bertmasking(nn.Module):
    def __init__(self, num_classes, num_clusters=256, chunk_size=500, dropout=0.15, 
                 hidden_dim=1536, n_masked_patch=5):
        super().__init__()
        # self.vit_pretrained = ...
        
        self.n_masked_patch = n_masked_patch
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_clusters = num_clusters
        self.chunk_size = chunk_size
        
        # [REMOVED] self.importance_gate = nn.Linear(hidden_dim, 1) 
        # BERT-style random masking does not need a learnable gate.

        self.cluster_pool = ClusterTokensPooling(num_clusters=num_clusters, dim=hidden_dim, heads=8)
        
        self.dropout = dropout
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True, dropout=dropout)
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, feats, attn_mask, epoch, stride=1):  # [B, T, hidden_dim]
        B, T, D = feats.shape
        
        # Strided sampling
        idxs = (torch.arange(0, T, stride, device=feats.device) + (epoch % stride)) % T
        feats = feats[:, idxs]  # [B, T//stride, D]
        attn_mask = attn_mask[:, idxs]  # [B, T//stride]
        
        # Recalculate T after striding
        T_new = feats.shape[1] 

        if self.n_masked_patch > 0 and self.training:
            # 1. Generate Random Scores (Uniform distribution)
            # This replaces the importance_gate logic
            rand_scores = torch.rand(B, T_new, device=feats.device)

            # 2. Mask Padding
            # Set scores of padding tokens to -inf so they are never selected for masking
            low_val = -torch.finfo(rand_scores.dtype).max / 2
            rand_scores = rand_scores.masked_fill(attn_mask == 0, low_val)

            # 3. Find Top-K Random Indices
            # We select the 'n_masked_patch' tokens with the highest random scores
            n_mask = min(self.n_masked_patch, T_new)
            _, to_mask_indices = torch.topk(rand_scores, n_mask, dim=-1)

            # 4. Create Mask
            # Create a mask of 1s, same dtype as feats (to avoid overflow issues)
            m_mask = torch.ones_like(attn_mask, dtype=feats.dtype) 
            
            # Scatter 0.0 to the selected random indices
            m_mask.scatter_(1, to_mask_indices, 0.0)
            
            # 5. Apply Mask
            attn_mask = attn_mask * m_mask

        # ----> Cluster pooling (learnable)
        clustered_feats = self.cluster_pool(feats, attn_mask)  # [B, num_clusters, hidden_dim]

        cls = self.cls_token.expand(B, -1, -1)
        out = torch.cat([cls, clustered_feats], dim=1)  
        
        # ----> Temporal encoder
        out = self.temporal_encoder(out)  # [B, num_clusters, hidden_dim]

        # ----> Pool cluster tokens
        cls = out[:, 0]  # [B, hidden_dim]
        mean = out[:, 1:].mean(dim=1)
        out = cls * 0.8 + mean * 0.2  # [B, hidden_dim]

        # ----> Classification
        logits = self.fc(out)  # [B, num_classes]
        return logits
    
class VTransUnified(nn.Module):
    def __init__(self, num_classes, chunk_size=500, dropout=0.15, hidden_dim=1536, 
                 n_masked_patch=5, mask_drop=0.5, ratio=0.5, 
                 dual_branch=False, num_masked_views=4):
        super().__init__()
        
        # --- Configs ---
        self.n_masked_patch = n_masked_patch
        self.mask_drop = mask_drop
        self.dual_branch = dual_branch
        self.num_masked_views = num_masked_views
        
        # --- Architecture (Shared) ---
        self.importance_gate = nn.Linear(hidden_dim, 1)
        self.token_pooler = AdaptiveCrossAttentionPooling(dim=hidden_dim, ratio=ratio, heads=8)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True, dropout=dropout)
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def _forward_backend(self, feats, mask):
        """Shared weights for both clean and masked paths"""
        B = feats.shape[0]
        clustered_feats = self.token_pooler(feats, mask)
        
        cls = self.cls_token.expand(B, -1, -1)
        out = torch.cat([cls, clustered_feats], dim=1)  
        out = self.temporal_encoder(out)

        cls_out = out[:, 0]
        mean_out = out[:, 1:].mean(dim=1)
        final_embed = cls_out * 0.8 + mean_out * 0.2
        return self.fc(final_embed)

    def forward(self, feats, attn_mask, epoch, stride=1):
        B, T, D = feats.shape
        idxs = (torch.arange(0, T, stride, device=feats.device) + (epoch % stride)) % T
        feats = feats[:, idxs]
        attn_mask = attn_mask[:, idxs]

        # --- Evaluation ---
        if not self.training:
            return self._forward_backend(feats, attn_mask)

        # --- Training ---
        # 1. Identify what to mask (Importance)
        importance = self.importance_gate(feats).squeeze(-1) 
        low_val = -torch.finfo(importance.dtype).max / 2
        importance = importance.masked_fill(attn_mask == 0, low_val)
        
        current_T = feats.shape[1]
        n_mask = min(self.n_masked_patch, current_T)
        _, indices = torch.topk(importance, n_mask, dim=-1)

        # 2. Helper to apply mask
        def get_masked_logits():
            num_drop = int(n_mask * self.mask_drop)
            if num_drop <= 0: return self._forward_backend(feats, attn_mask)
            
            noise = torch.rand_like(indices.float())
            rand_select = torch.argsort(noise, dim=-1)[:, :num_drop]
            to_mask_indices = indices.gather(1, rand_select)
            
            m_mask = torch.ones_like(attn_mask, dtype=feats.dtype) 
            m_mask.scatter_(1, to_mask_indices, 0.0)
            return self._forward_backend(feats, attn_mask * m_mask)

        # --- Branch Logic ---
        if self.dual_branch:
            # Returns (Clean, [Masked...]) -> Triggers "Case A" in loop
            logits_clean = self._forward_backend(feats, attn_mask)
            masked_list = [get_masked_logits() for _ in range(self.num_masked_views)]
            return logits_clean, masked_list
        else:
            # Returns Single Masked Logits -> Triggers "Case B" in loop
            # This forces the model to learn from the hard view
            return get_masked_logits()