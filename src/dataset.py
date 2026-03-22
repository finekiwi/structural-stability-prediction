from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

LABEL_MAP = {"stable": 0, "unstable": 1}

INTERP_MAP = {
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "nearest": cv2.INTER_NEAREST,
    "lanczos": cv2.INTER_LANCZOS4,
}


def build_transforms(img_size: int, mode: str, preprocess_cfg: dict):
    """Build (geo_tfm, photo_tfm, final_tfm) for a given mode.

    geo_tfm:   A.ReplayCompose — applied to front, then replayed on top (shared params)
    photo_tfm: A.Compose      — applied independently to each view
    final_tfm: A.Compose      — Resize + Normalize + ToTensor (both views)
    """
    interp = INTERP_MAP.get(preprocess_cfg["interpolation"], cv2.INTER_LINEAR)
    mean = list(preprocess_cfg["mean"])
    std = list(preprocess_cfg["std"])

    if mode == "train":
        geo_tfm = A.ReplayCompose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=0,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.5,
            ),
        ])
        photo_tfm = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.GaussNoise(p=0.2),
        ])
    else:
        geo_tfm = A.ReplayCompose([])
        photo_tfm = A.Compose([])

    final_tfm = A.Compose([
        A.Resize(img_size, img_size, interpolation=interp),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    return geo_tfm, photo_tfm, final_tfm


class MultiViewDataset(Dataset):
    """Dataset returning (front_tensor, top_tensor, label) for train/val or
    (front_tensor, top_tensor) for test."""

    def __init__(
        self,
        df: pd.DataFrame,
        data_dir: str,
        geo_tfm,
        photo_tfm,
        final_tfm,
        mode: str = "train",
    ):
        self.df = df.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.geo_tfm = geo_tfm
        self.photo_tfm = photo_tfm
        self.final_tfm = final_tfm
        self.mode = mode

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, path: Path) -> np.ndarray:
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        sample_id = row["id"]

        front_img = self._load_image(self.data_dir / sample_id / "front.png")
        top_img = self._load_image(self.data_dir / sample_id / "top.png")

        # 1. Shared geometric aug (same params for both views via ReplayCompose)
        result = self.geo_tfm(image=front_img)
        front_geo = result["image"]
        top_geo = A.ReplayCompose.replay(result["replay"], image=top_img)["image"]

        # 2. Independent photometric aug + normalize/resize
        front_out = self.final_tfm(image=self.photo_tfm(image=front_geo)["image"])["image"]
        top_out = self.final_tfm(image=self.photo_tfm(image=top_geo)["image"])["image"]

        if self.mode == "test":
            return front_out, top_out

        label = torch.tensor(LABEL_MAP[row["label"]], dtype=torch.float32)
        return front_out, top_out, label
