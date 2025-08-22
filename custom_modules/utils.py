from typing import List

import torch.nn as nn


class FeatureSelector(nn.Module):
    def __init__(self, feature_index: int, out_channels: int):
        super().__init__()
        self.feature_index = feature_index
        self.out_channels = out_channels  # ⭐ 明确告诉框架输出通道数

    def forward(self, features: List):
        return features[self.feature_index]
