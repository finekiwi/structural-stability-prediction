"""Training script for structural stability prediction.

Usage:
    python -m src.train --config configs/baseline.yaml
    # fold: all  -> runs all 5 folds, saves OOF predictions
    # fold: 0    -> runs single fold
"""

import argparse
import random
import os
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
import yaml
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import MultiViewDataset, build_transforms
from src.model import build_model, build_model_from_spec


def load_config(path):
    with open(path) as f:
        d = yaml.safe_load(f)
    return SimpleNamespace(**d)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_loader(df, data_dir, preprocess_cfg, img_size, mode, batch_size, num_workers, seed):
    geo_tfm, photo_tfm, final_tfm = build_transforms(img_size, mode, preprocess_cfg)
    ds = MultiViewDataset(df, data_dir, geo_tfm, photo_tfm, final_tfm, mode=mode)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(mode == "train"),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        worker_init_fn=worker_init_fn,
        generator=generator,
    )


def build_scheduler(optimizer, cfg):
    if not (0 <= cfg.warmup_epochs < cfg.epochs):
        raise ValueError(
            f"warmup_epochs={cfg.warmup_epochs} must be in [0, epochs={cfg.epochs})"
        )
    if cfg.warmup_epochs == 0:
        return CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    return SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=cfg.warmup_start_factor, total_iters=cfg.warmup_epochs),
            CosineAnnealingLR(optimizer, T_max=cfg.epochs - cfg.warmup_epochs),
        ],
        milestones=[cfg.warmup_epochs],
    )


def smooth_labels(target, alpha):
    return target * (1.0 - alpha) + 0.5 * alpha


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for front, top, label in loader:
        front, top = front.to(device), top.to(device)
        logits = model(front, top).squeeze(1)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(label.numpy().tolist())
    probs_arr = np.array(all_probs)
    labels_arr = np.array(all_labels)
    logloss = log_loss(labels_arr, probs_arr, labels=[0, 1])
    acc = ((probs_arr >= 0.5).astype(int) == labels_arr.astype(int)).mean()
    return logloss, acc, probs_arr


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip, label_smoothing, ema=None):
    model.train()
    total_loss = 0.0
    for front, top, label in tqdm(loader, leave=False):
        front, top, label = front.to(device), top.to(device), label.to(device)
        if label_smoothing > 0:
            label = smooth_labels(label, label_smoothing)
        optimizer.zero_grad()
        logits = model(front, top).squeeze(1)
        loss = criterion(logits, label)
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if ema is not None:
            ema.update(model)
        total_loss += loss.item() * front.size(0)
    return total_loss / len(loader.dataset)


def train_fold(fold_idx, train_df, val_df, cfg, data_dir, preprocess_cfg, ckpt_dir, device):
    """Train a single fold. Returns (val_probs, best_logloss)."""
    set_seed(cfg.seed + fold_idx)

    model, _ = build_model_from_spec(
        backbone_name=cfg.backbone,
        fusion=cfg.fusion,
        dropout=cfg.dropout,
        pretrained=cfg.pretrained,
    )
    model = model.to(device)

    train_dir = data_dir / "train"
    train_loader = build_loader(train_df, train_dir, preprocess_cfg, cfg.img_size, "train", cfg.batch_size, cfg.num_workers, cfg.seed)
    val_loader = build_loader(val_df, train_dir, preprocess_cfg, cfg.img_size, "val", cfg.batch_size, cfg.num_workers, cfg.seed)

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = build_scheduler(optimizer, cfg)
    criterion = nn.BCEWithLogitsLoss()

    ema = None
    if cfg.use_ema:
        from timm.utils import ModelEmaV3
        ema = ModelEmaV3(model, decay=cfg.ema_decay)

    best_logloss = float("inf")
    best_probs = None
    patience_counter = 0

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            cfg.grad_clip, cfg.label_smoothing, ema,
        )

        eval_model = ema.module if (ema is not None) else model
        val_logloss, val_acc, val_probs = evaluate(eval_model, val_loader, device)

        current_lr = scheduler.get_last_lr()[0]
        print(
            f"  Epoch {epoch:03d}/{cfg.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_logloss={val_logloss:.6f} | "
            f"val_acc={val_acc:.4f} | "
            f"lr={current_lr:.2e}"
        )
        wandb.log({
            f"fold{fold_idx}/epoch": epoch,
            f"fold{fold_idx}/train_loss": train_loss,
            f"fold{fold_idx}/val_logloss": val_logloss,
            f"fold{fold_idx}/val_acc": val_acc,
        })

        if val_logloss < best_logloss:
            best_logloss = val_logloss
            best_probs = val_probs
            patience_counter = 0
            state_dict = ema.module.state_dict() if ema is not None else model.state_dict()
            torch.save({
                "model_state_dict": state_dict,
                "model_cfg": {
                    "backbone_name": cfg.backbone,
                    "fusion": cfg.fusion,
                    "dropout": cfg.dropout,
                },
                "preprocess_cfg": preprocess_cfg,
                "val_logloss": best_logloss,
                "epoch": epoch,
            }, ckpt_dir / "best.pt")
            print(f"    -> Saved best (val_logloss={best_logloss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stop_patience:
                print(f"  Early stopping at epoch {epoch}")
                break

        scheduler.step()

    return best_probs, best_logloss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    _, preprocess_cfg = build_model(cfg)

    data_dir = Path(cfg.data_dir)
    train_df = pd.read_csv(data_dir / "train.csv")

    from src.dataset import LABEL_MAP
    labels = train_df["label"].map(LABEL_MAP).values

    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    splits = list(skf.split(train_df, labels))

    fold_list = list(range(cfg.n_folds)) if str(cfg.fold) == "all" else [int(cfg.fold)]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg.output_root) / f"{cfg.wandb_run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run_dir] {run_dir}")

    wandb.init(
        project=cfg.wandb_project,
        name=cfg.wandb_run_name,
        config=vars(cfg),
        mode=cfg.wandb_mode,
    )

    oof_probs = np.zeros(len(train_df))
    fold_scores = []

    for fold_idx in fold_list:
        print(f"\n{'='*50}")
        print(f"Fold {fold_idx + 1}/{cfg.n_folds}")
        print(f"{'='*50}")

        train_idx, val_idx = splits[fold_idx]
        fold_train_df = train_df.iloc[train_idx]
        fold_val_df = train_df.iloc[val_idx]

        ckpt_dir = run_dir / f"fold_{fold_idx}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        val_probs, best_logloss = train_fold(
            fold_idx, fold_train_df, fold_val_df,
            cfg, data_dir, preprocess_cfg, ckpt_dir, device,
        )

        oof_probs[val_idx] = val_probs
        fold_scores.append(best_logloss)
        print(f"Fold {fold_idx} best val_logloss: {best_logloss:.6f}")
        wandb.log({f"fold{fold_idx}/best_val_logloss": best_logloss})

    if str(cfg.fold) == "all":
        oof_logloss = log_loss(labels, oof_probs, labels=[0, 1])
        oof_acc = ((oof_probs >= 0.5).astype(int) == labels).mean()
        print(f"\n{'='*50}")
        print(f"OOF CV logloss : {oof_logloss:.6f}")
        print(f"OOF CV accuracy: {oof_acc:.4f}")
        for i, s in enumerate(fold_scores):
            print(f"  fold {i}: {s:.6f}")
        print(f"{'='*50}")
        wandb.log({"oof_logloss": oof_logloss, "oof_acc": oof_acc})

        oof_df = train_df[["id"]].copy()
        oof_df["prob"] = oof_probs
        oof_df["label"] = train_df["label"]
        oof_df.to_csv(run_dir / "oof_preds.csv", index=False)
        print(f"OOF predictions saved: {run_dir / 'oof_preds.csv'}")

    wandb.finish()


if __name__ == "__main__":
    main()
