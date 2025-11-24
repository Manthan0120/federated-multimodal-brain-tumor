import torch.nn as nn
from .backbone import MobileNetV3Backbone
from .local_heads import RandomForestHead


class MultimodalFLModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.body = MobileNetV3Backbone(
            pretrained=config["model"]["pretrained"],
            freeze=config["model"]["freeze_body"]
        )
        self.feature_dim = self.body.feature_dim
        self.local_head = None  # Set per client

    def forward(self, x):
        return self.body(x)

    def set_local_head(self, head):
        self.local_head = head

    def get_gradient_params(self):
        return self.body.parameters()

    def get_features(self, x):
        return self.body(x)