from xml.parsers.expat import model
from tqdm import tqdm
import torch
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassAUROC
)

import os
import pandas as pd
import torch.optim as optim 
import torch.nn as nn
import numpy as np
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from .utils import EarlyStopping


def get_optim(model, args):
	if args.opt == "adam":
		optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
	elif args.opt == 'sgd':
		optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
	else:
		raise NotImplementedError
	return optimizer

def build_metrics(num_classes, device):
    return {
        "acc": MulticlassAccuracy(
            num_classes=num_classes
        ).to(device),
        "f1": MulticlassF1Score(
            num_classes=num_classes,
            average="macro"
        ).to(device),

        "precision": MulticlassPrecision(
            num_classes=num_classes,
            average="macro"
        ).to(device),

        "recall": MulticlassRecall(
            num_classes=num_classes,
            average="macro"
        ).to(device),

        "auc": MulticlassAUROC(
            num_classes=num_classes,
            average="macro"
        ).to(device),
    }


def get_label_smoothing(epoch, max_epochs):
    warmup = int(0.3 * max_epochs)
    cooldown = int(0.7 * max_epochs)

    if epoch < warmup:
        return 0.1
    elif epoch < cooldown:
        return 0.05
    else:
        return 0.05
    
    
def train(run, logger, args, train_loader, val_loader, device, model = None):
    writer_dir = args.results_dir_path
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)
    
    if model is not None:
        _ = model.to(device)
    else:
        raise ValueError

    optimizer = get_optim(model, args)
    
    base_lr = optimizer.param_groups[0]['lr']
    warmup_epochs = 10
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6
    )   
    
    scaler = GradScaler()

    early_stopping = EarlyStopping(
        patience=args.patience, 
        verbose=True, 
        path=os.path.join(writer_dir, 'best_val_loss_checkpoint.pth')
    )
    
    
    metrics = build_metrics(args.num_classes, device)
    
    best_acc = -1
    try: 
        for epoch in range(0, args.max_epochs):
            # MASK_START_EPOCH = 15 # Huy
            # if epoch < warmup_epochs:
            #     current_lr = base_lr * ((epoch + 1) / warmup_epochs)
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = current_lr
            # if epoch == MASK_START_EPOCH: # Huy
            #     logger.info(f"🚀 Masking Phase Started. Performing SOFT Reset.")
            #     new_lr = base_lr * 0.1 
                
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = new_lr 
            #     scheduler._reset() 
            #     scheduler.best = float('inf')
            ls = get_label_smoothing(epoch, args.max_epochs)
            loss_fn = nn.CrossEntropyLoss(label_smoothing=ls)
            val_loss_fn = nn.CrossEntropyLoss()
            train_loss = train_loop(run, scaler, logger, epoch, model, train_loader, metrics, optimizer, device, 1, loss_fn)
            acc, val_loss = validate(run, logger, epoch, model, val_loader, device, args.strategy, metrics, 1, val_loss_fn, args.results_dir_path)
            # if epoch < MASK_START_EPOCH + 5:  # The "Buffer" # Huy
            #     pass  # Don't let the scheduler see the crash!
            # else:
            #     scheduler.step(val_loss)
            scheduler.step(val_loss) 
            
            # eval_table.add_data(epoch, res["accuracy"], res["f1_score"], res["precision"], res["recall"])
            # if args.early_stopping:
            #     model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
            # else:
            #     torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))
            
            if acc > best_acc:
                best_acc = acc
                
                # --- CHANGE STARTS HERE ---
                
                # Check if a specific save_name was passed, otherwise default to "best_model"
                # We use getattr to be safe if the key doesn't exist
                name_prefix = getattr(args, "save_name", "best_model")
                
                # Create a filename that won't be overwritten
                filename = 'new/' + f"{name_prefix}.pth"
                save_path = os.path.join(writer_dir, filename)
                
                # --- CHANGE ENDS HERE ---

                torch.save(model.state_dict(), save_path)
                logger.info(f"Saved new best accuracy model: {save_path} | Acc: {acc:.4f}")
                
            if epoch >= warmup_epochs:
                
                # Step Scheduler
                scheduler.step(val_loss) 

                # Check Early Stopping
                early_stopping(val_loss, model)
                
                if early_stopping.early_stop:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
            
            
            # if acc > best_acc:
            #     best_acc = acc
            #     torch.save(model.state_dict(), os.path.join(f"results/best_model_{epoch}.pth"))
    except Exception as e:
        logger.error(e)


def train_loop(run, scaler, logger, epoch, model, loader, metrics, optimizer, device, stride = 10, loss_fn = None):   
    model.train()
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    for m in metrics.values():
        m.reset()
        
    print(f'==============Training Epoch {epoch}===============')
    for step, (feats, labels, attn_mask) in enumerate(loader):
        feats = feats.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        attn_mask = attn_mask.to(device, non_blocking=True)

        # with autocast(dtype=torch.float16):
        outputs = model(feats, attn_mask, epoch, stride)
        # MASK_WARMUP_EPOCHS = 15
        # 2. Check if the model returned a tuple (Dual Branch) or Tensor (Single Branch)
        # if isinstance(outputs, tuple): # Huy
        #     logits_clean, logits_masked_list = outputs
            
        #     # --- PRIMARY LOSS (Clean Branch vs Label) ---
        #     # This is the "Teacher". It learns from the ground truth.
        #     loss_clean = loss_fn(logits_clean, labels)
            
        #     # --- AUXILIARY LOSS (Consistency: Masked vs Clean) ---
        #     # Don't use 'labels' here. Use 'logits_clean' as the target.
            
        #     # CRITICAL: Detach logits_clean! 
        #     # We want Masked to chase Clean. We DO NOT want Clean to become dumber to match Masked.
        #     target_logits = logits_clean.detach()
            
        #     # Use MSE (Mean Squared Error) to force the distributions to align
        #     consistency_criterion = torch.nn.MSELoss()
            
        #     mask_losses = []
        #     for l_masked in logits_masked_list:
        #         # Calculate how far the masked prediction is from the clean prediction
        #         mask_losses.append(consistency_criterion(l_masked, target_logits))
                
        #     loss_consistency = sum(mask_losses) / len(mask_losses)
            
        #     # Combine
        #     if epoch < MASK_WARMUP_EPOCHS:
        #         loss = loss_clean
        #     else:
        #         # You can use a higher weight now because MSE is softer than CE
        #         loss = loss_clean + (5.0 * loss_consistency)

        #     logits = logits_clean
        # else:
        #     # Standard forward pass (Validation or masking off)
        logits = outputs
        loss = loss_fn(logits, labels)
        total_loss += loss.item()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # logits = torch.mean(torch.stack(logits), dim=0)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        
        
        metrics["acc"].update(preds, labels)
        metrics["f1"].update(preds, labels)
        metrics["precision"].update(preds, labels)
        metrics["recall"].update(preds, labels)
        metrics["auc"].update(probs, labels)
        
        
        if step % 50 == 0:
            logger.info(
                f"Epoch {epoch} [{step}/{len(loader)}] "
                f"Loss: {loss.item():.4f}"
            )

    
    acc = metrics["acc"].compute().item()
    f1 = metrics["f1"].compute().item()
    precision = metrics["precision"].compute().item()
    recall = metrics["recall"].compute().item()
    auc = metrics["auc"].compute().item()
    
    # calculate loss and error for epoch
    avg_loss = total_loss / len(loader)
    print(loss)
    run.log({
        "train/loss": avg_loss,
        "train/accuracy": acc,
        "train/f1": f1,
        "train/precision": precision,
        "train/recall": recall,
        "train/auc": auc,
        "epoch": epoch
    })
    # run.log({"train/loss": avg_loss, "epoch": epoch})
    return avg_loss

@torch.no_grad()
def validate(run, logger,  epoch, model, loader, device , strategy, metrics , stride = 1,loss_fn = None, results_dir=None):
    model.eval()
    total_loss = 0.0
    
    # reset metric states
    for m in metrics.values():
        m.reset()
        
    offsets = list(range(stride))

    for feats, labels, attn_mask in loader:
        feats = feats.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        attn_mask = attn_mask.to(device, non_blocking=True)

        
        #predict
        logits_all = []
        loss = 0.0
        for e in offsets:
            logits_e = model(feats, attn_mask, e, stride)
            loss += loss_fn(logits_e, labels) / len(offsets)
            logits_all.append(logits_e)
        
        logits = torch.mean(torch.stack(logits_all), dim=0)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        total_loss += loss.item()
        
        metrics["acc"].update(preds, labels)
        metrics["f1"].update(preds, labels)
        metrics["precision"].update(preds, labels)
        metrics["recall"].update(preds, labels)
        metrics["auc"].update(probs, labels)
    
    
    val_loss = total_loss / len(loader)
    
    acc = metrics["acc"].compute().item()
    f1 = metrics["f1"].compute().item()
    precision = metrics["precision"].compute().item()
    recall = metrics["recall"].compute().item()
    auc = metrics["auc"].compute().item()
    
    
    run.log({
        "val/loss": val_loss,
        "val/accuracy": acc,
        "val/f1": f1,
        "val/precision": precision,
        "val/recall": recall,
        "val/auc": auc,
        "epoch": epoch
    })
    
    logger.info(
        f"[VAL] Epoch {epoch} | Loss: {val_loss:.4f} | Acc: {acc:.4f}"
    )
    return acc, val_loss

@torch.no_grad()
def test(run, logger, args, model, test_loader, device, metrics, stride=1):
    """
    Evaluates the best model on the test set using multi-view testing.
    """
    model.eval()
    for m in metrics.values():
        m.reset()

    # Lists for sklearn reporting
    all_preds = []
    all_targets = []
    all_probs = []

    # Define offsets for multi-view evaluation (same logic as your validate function)
    # If stride > 1, this ensembles predictions from shifted temporal windows
    offsets = list(range(stride))
    
    logger.info(f"Starting Test Evaluation with {len(offsets)} views (stride={stride})...")
    
    for feats, labels, attn_mask in tqdm(test_loader, desc="Testing"):
        feats = feats.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        attn_mask = attn_mask.to(device, non_blocking=True)

        logits_all = []
        for e in offsets:
            # We pass 'e' as the 'epoch' argument to shift the sampling window
            logits_e = model(feats, attn_mask, e, stride)
            logits_all.append(logits_e)
        
        # Ensemble: Average logits across all views
        logits = torch.mean(torch.stack(logits_all), dim=0)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        # Update TorchMetrics
        metrics["acc"].update(preds, labels)
        metrics["f1"].update(preds, labels)
        metrics["precision"].update(preds, labels)
        metrics["recall"].update(preds, labels)
        metrics["auc"].update(probs, labels)

        # Store for Sklearn/Analysis
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    # --- Compute Final TorchMetrics ---
    acc = metrics["acc"].compute().item()
    f1 = metrics["f1"].compute().item()
    precision = metrics["precision"].compute().item()
    recall = metrics["recall"].compute().item()
    auc = metrics["auc"].compute().item()

    logger.info(f"Test Results | Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
    
    run.log({
            "test/accuracy": acc,
            "test/f1": f1,
            "test/precision": precision,
            "test/recall": recall,
            "test/auc": auc
        })