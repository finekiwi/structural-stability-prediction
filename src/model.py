import timm
import torch
import torch.nn as nn
from timm.data import resolve_data_config


class MultiViewModel(nn.Module):
    """Two-view model: shared backbone + concat fusion + classifier head."""

    def __init__(self, backbone: nn.Module, fusion: str = "concat", dropout: float = 0.3):
        super().__init__()
        self.backbone = backbone
        self.fusion = fusion
        in_features = backbone.num_features

        if fusion == "concat":
            self.classifier = nn.Sequential(
                nn.Linear(in_features * 2, 512),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(512, 1),
            )
        else:
            raise ValueError(f"Unknown fusion: {fusion!r}. Supported: 'concat'")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, front: torch.Tensor, top: torch.Tensor) -> torch.Tensor:
        f_front = self.encode(front)
        f_top = self.encode(top)
        if self.fusion == "concat":
            fused = torch.cat([f_front, f_top], dim=1)
        return self.classifier(fused)


def build_model_from_spec(
    backbone_name: str,
    fusion: str,
    dropout: float,
    pretrained: bool = True,
):
    """Single entry point for model creation used by both train and inference.

    Returns:
        model: MultiViewModel instance
        preprocess_cfg: dict with keys img_size (None, set by caller), mean, std, interpolation
    """
    backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
    timm_data_cfg = resolve_data_config(pretrained_cfg=backbone.pretrained_cfg, model=backbone)
    preprocess_cfg = {
        "img_size": None,  # caller fills this in
        "mean": timm_data_cfg["mean"],
        "std": timm_data_cfg["std"],
        "interpolation": timm_data_cfg["interpolation"],
    }
    model = MultiViewModel(backbone=backbone, fusion=fusion, dropout=dropout)
    return model, preprocess_cfg


def build_model(cfg):
    """Convenience wrapper for training: fills img_size from cfg."""
    model, preprocess_cfg = build_model_from_spec(
        backbone_name=cfg.backbone,
        fusion=cfg.fusion,
        dropout=cfg.dropout,
        pretrained=cfg.pretrained,
    )
    preprocess_cfg["img_size"] = cfg.img_size
    return model, preprocess_cfg
