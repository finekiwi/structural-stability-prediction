"""5-fold ensemble inference + dev evaluation.

Usage:
    python -m src.inference --run-dir outputs/<run_name> [--eval-dev]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import MultiViewDataset, build_transforms
from src.model import build_model_from_spec


def build_loader(df, data_dir, preprocess_cfg, mode, batch_size, num_workers):
    geo_tfm, photo_tfm, final_tfm = build_transforms(preprocess_cfg["img_size"], mode, preprocess_cfg)
    ds = MultiViewDataset(df, data_dir, geo_tfm, photo_tfm, final_tfm, mode=mode)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )


@torch.no_grad()
def predict(model, loader, device, has_label=False):
    model.eval()
    all_probs, all_labels = [], []
    for batch in tqdm(loader, leave=False):
        if has_label:
            front, top, label = batch
            all_labels.extend(label.numpy().tolist())
        else:
            front, top = batch
        front, top = front.to(device), top.to(device)
        logits = model(front, top).squeeze(1)
        all_probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    return np.array(all_probs), np.array(all_labels) if has_label else None


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model, _ = build_model_from_spec(**ckpt["model_cfg"], pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    return model, ckpt["preprocess_cfg"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--eval-dev", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")
    print(f"Run dir: {run_dir}")

    ckpt_paths = sorted(run_dir.glob("fold_*/best.pt"))
    if not ckpt_paths:
        raise FileNotFoundError(f"No checkpoints found in {run_dir}/fold_*/best.pt")
    print(f"Found {len(ckpt_paths)} fold checkpoints: {[p.parent.name for p in ckpt_paths]}")

    for candidate in [
        Path("/kaggle/input/datasets/yunaism/structural-stability-prediction"),
        Path("data"),
    ]:
        if (candidate / "train.csv").exists():
            data_dir = candidate
            break
    else:
        raise FileNotFoundError("Cannot find data directory")

    print(f"Data dir: {data_dir}")

    # ── Dev evaluation ────────────────────────────────────────────────────────
    if args.eval_dev:
        dev_df = pd.read_csv(data_dir / "dev.csv")
        dev_dir = data_dir / "dev"
        LABEL_MAP = {"stable": 0, "unstable": 1}
        dev_labels = dev_df["label"].map(LABEL_MAP).values

        fold_probs_dev = []
        for ckpt_path in ckpt_paths:
            model, pcfg = load_model(ckpt_path, device)
            loader = build_loader(dev_df, dev_dir, pcfg, "val", args.batch_size, args.num_workers)
            probs, _ = predict(model, loader, device, has_label=True)
            fold_probs_dev.append(probs)
            print(f"  {ckpt_path.parent.name} dev logloss: {log_loss(dev_labels, probs, labels=[0, 1]):.6f}")

        ensemble_probs_dev = np.mean(fold_probs_dev, axis=0)
        dev_logloss = log_loss(dev_labels, ensemble_probs_dev, labels=[0, 1])
        dev_acc = ((ensemble_probs_dev >= 0.5).astype(int) == dev_labels).mean()
        print(f"\n[Dev Ensemble] logloss={dev_logloss:.6f}  acc={dev_acc:.4f}")

    # ── Test inference ────────────────────────────────────────────────────────
    test_df = pd.read_csv(data_dir / "sample_submission.csv")[["id"]]
    test_dir = data_dir / "test"

    fold_probs_test = []
    for ckpt_path in ckpt_paths:
        model, pcfg = load_model(ckpt_path, device)
        loader = build_loader(test_df, test_dir, pcfg, "test", args.batch_size, args.num_workers)
        probs, _ = predict(model, loader, device, has_label=False)
        fold_probs_test.append(probs)

    ensemble_probs = np.mean(fold_probs_test, axis=0)

    submission = pd.DataFrame({
        "id": test_df["id"].values,
        "unstable_prob": ensemble_probs,
        "stable_prob": 1.0 - ensemble_probs,
    })

    out_path = run_dir / "submission.csv"
    submission.to_csv(out_path, index=False)
    print(f"\nSubmission saved: {out_path}")
    print(f"Shape: {submission.shape}")
    assert ((submission["unstable_prob"] + submission["stable_prob"] - 1.0).abs() < 1e-6).all()
    print("Sum check: OK")


if __name__ == "__main__":
    main()
