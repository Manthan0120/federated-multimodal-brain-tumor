# src/models/multimodal.py
import torch
import torch.nn as nn
from typing import Optional
from .backbone import MobileNetV3Backbone  # ← Correct relative import


class MultimodalFLModel(nn.Module):
    """
    Global shared feature extractor only.
    Local classifier heads are attached per-client (RF, DT, MLP, etc.)
    """
    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        model_cfg = config["model"]

        self.body = MobileNetV3Backbone(
            pretrained=model_cfg.get("pretrained", True),
            freeze=model_cfg.get("freeze_body", True)  # ← Enforced from config
        )

        self.feature_dim = self.body.feature_dim  # 576
        self.local_head = None  # Set by client

    def forward(self, x):
        """Convenience: same as get_features()"""
        return self.body(x)

    def get_features(self, x):
        """Main method used during inference and head training"""
        return self.body(x)

    def set_local_head(self, head):
        """Attach a local classifier (e.g. RandomForestHead)"""
        self.local_head = head

    def get_gradient_params(self):
        """
        Only backbone parameters are sent/received in FedAvg.
        Local heads are private and never shared.
        """
        return list(self.body.parameters())