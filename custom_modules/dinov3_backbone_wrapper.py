import torch
import torch.nn as nn
from dinov3_backbone import DINOv3ConvNeXtTinyBackbone


class DINOBackboneYOLOWrapper(nn.Module):
    """
    将 DINOv3ConvNeXtTinyBackbone 输出通道映射到 YOLO11 head 期望通道：
      P3: 192 -> 256
      P4: 384 -> 512
      P5: 768 -> 1024.
    """

    def __init__(self, repo_path: str, weight_path: str, freeze: bool = False):
        super().__init__()
        # 初始化 DINOv3 backbone
        self.backbone = DINOv3ConvNeXtTinyBackbone(repo_path=repo_path, weight_path=weight_path, freeze=freeze)

        # YOLO11 head 期望的输出通道
        self.out_channels = [256, 512, 1024]

        # 1x1 conv 映射通道
        self.p3_conv = nn.Conv2d(self.backbone.out_channels[0], self.out_channels[0], kernel_size=1)
        self.p4_conv = nn.Conv2d(self.backbone.out_channels[1], self.out_channels[1], kernel_size=1)
        self.p5_conv = nn.Conv2d(self.backbone.out_channels[2], self.out_channels[2], kernel_size=1)

    def forward(self, x: torch.Tensor):
        # 获取 DINO backbone 特征
        P3, P4, P5 = self.backbone(x)

        # 1x1 conv 映射到 YOLO head 期望通道
        P3 = self.p3_conv(P3)
        P4 = self.p4_conv(P4)
        P5 = self.p5_conv(P5)

        # 返回 tuple，方便 YOLO head 使用
        return (P3, P4, P5)
