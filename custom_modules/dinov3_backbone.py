# custom_modules/dinov3_backbone.py
# 最终重构版: 直接调用官方 get_intermediate_layers API

import torch
import torch.nn as nn
from typing import List
import os


class DINOv3ConvNeXtTinyBackbone(nn.Module):
    """
    用于YOLO的DINOv3 ConvNeXt-Tiny骨干网络封装类。
    (最终重构版: 直接调用官方 get_intermediate_layers API)
    """

    def __init__(self, repo_path: str, weight_path: str, freeze: bool = False):
        super().__init__()

        if not os.path.isdir(repo_path):
            raise FileNotFoundError(f"DINOv3 repository not found at: {repo_path}")
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Weight file not found at: {weight_path}")

        # 加载完整的DINOv3模型并将其保存为类的一个属性
        self.model = torch.hub.load(
            repo_or_dir=repo_path,
            model='dinov3_convnext_tiny',
            source='local',
            weights=weight_path
        )

        # 确定输出通道数
        self._out_channels = [192, 384, 768]  # P3, P4, P5

        if freeze:
            print("Freezing DINOv3 ConvNeXt-Tiny backbone.")
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        通过调用官方的 get_intermediate_layers API 来执行前向传播。
        """
        # 直接调用官方函数提取 P3, P4, P5 特征
        # n=[1, 2, 3] -> 提取 stage 1, 2, 3 的输出
        # reshape=True -> 确保输出是 [B, C, H, W] 的二维特征图格式
        features_tuple = self.model.get_intermediate_layers(x, n=[1, 2, 3], reshape=True)

        # 官方函数返回的是元组(Tuple)，我们将其转换为列表(List)以保持接口一致
        return list(features_tuple)

    @property
    def out_channels(self):
        return self._out_channels


# --- 测试代码保持不变，用于验证重构后的类是否依然工作正常 ---
# if __name__ == '__main__':
#     DINOV3_REPO_PATH = "../dinov3-main"
#     LOCAL_WEIGHT_PATH = "../dinov3-main/checkpoints/dinov3_convnext_tiny.pth"
#
#     backbone = DINOv3ConvNeXtTinyBackbone(
#         repo_path=DINOV3_REPO_PATH,
#         weight_path=LOCAL_WEIGHT_PATH
#     )
#     backbone.eval()
#
#     dummy_input = torch.randn(1, 3, 640, 640)
#
#     with torch.no_grad():
#         output_features = backbone(dummy_input)
#
#     print("\nBackbone output feature shapes (after refactoring):")
#     for i, feature in enumerate(output_features):
#         print(f"  Feature {i + 1} (P{i + 3}): {feature.shape}")
#
#     print(f"\nBackbone out_channels property: {backbone.out_channels}")

# if __name__ == '__main__':
#     # --- 步骤1: 保持不变，加载DINOv3 Backbone并提取原始特征 ---
#     DINOV3_REPO_PATH = "../dinov3-main"
#     LOCAL_WEIGHT_PATH = "../dinov3-main/checkpoints/dinov3_convnext_tiny.pth"
#
#     # 实例化我们的Backbone
#     backbone = DINOv3ConvNeXtTinyBackbone(
#         repo_path=DINOV3_REPO_PATH,
#         weight_path=LOCAL_WEIGHT_PATH
#     )
#     backbone.eval()
#
#     # 创建一个虚拟输入图像
#     dummy_input = torch.randn(1, 3, 640, 640)
#
#     # 执行前向传播，得到原始特征
#     with torch.no_grad():
#         output_features = backbone(dummy_input)
#
#     print("\nBackbone's RAW output feature shapes:")
#     for i, feature in enumerate(output_features):
#         print(f"  Raw Feature {i + 1} (P{i + 3}): {feature.shape}")
#
#     # ----------------------------------------------------------------
#     # --- 步骤2: 新增代码，模拟YAML中的Conv适配器层 ---
#     # ----------------------------------------------------------------
#     print("\n--- Simulating Conv Adapters (as specified in your YAML) ---")
#
#     # 根据您的YAML定义，创建三个1x1的卷积适配器
#     # P3适配器: 输入192通道, 输出256通道
#     p3_adapter = torch.nn.Conv2d(in_channels=192, out_channels=256, kernel_size=1)
#
#     # P4适配器: 输入384通道, 输出512通道
#     p4_adapter = torch.nn.Conv2d(in_channels=384, out_channels=512, kernel_size=1)
#
#     # P5适配器: 输入768通道, 输出1024通道
#     p5_adapter = torch.nn.Conv2d(in_channels=768, out_channels=1024, kernel_size=1)
#
#     print("Conv adapters created successfully.")
#
#     # 从backbone的输出中分离出P3, P4, P5的原始特征
#     p3_raw_feature, p4_raw_feature, p5_raw_feature = output_features
#
#     # 将原始特征分别通过对应的适配器
#     with torch.no_grad():
#         p3_adapted = p3_adapter(p3_raw_feature)
#         p4_adapted = p4_adapter(p4_raw_feature)
#         p5_adapted = p5_adapter(p5_raw_feature)
#
#     # --- 步骤3: 打印经过适配器处理后的最终特征形状 ---
#     print("\n--- Final feature shapes AFTER Conv Adapters ---")
#     print(f"Adapted P3 feature shape: {p3_adapted.shape}")
#     print(f"Adapted P4 feature shape: {p4_adapted.shape}")
#     print(f"Adapted P5 feature shape: {p5_adapted.shape}")
#
#     from ultralytics.nn.modules import C2PSA,SPPF
#
#     # --- 步骤3: 模拟 SPPF 操作 ---
#     print("\n--- Simulating SPPF layer ---")
#     # 根据YAML定义 [-1, 1, SPPF, [1024, 5]]
#     sppf_layer = SPPF(c1=1024, c2=1024, k=5)
#
#     with torch.no_grad():
#         sppf_output = sppf_layer(p5_adapted)
#
#     print(f"Feature shape AFTER SPPF: {sppf_output.shape}")
#
#     # ----------------------------------------------------------------
#     # --- 步骤4: 新增代码，模拟 C2PSA 操作 ---
#     # ----------------------------------------------------------------
#     print("\n--- Simulating C2PSA layer ---")
#
#     # 根据YAML定义 [-1, 2, C2PSA, [1024]]
#     # 假设 C2PSA 的构造函数接收输入通道 c1 和输出通道 c2
#     # 在这里，输入和输出通道都是 1024
#     c2psa_layer = C2PSA(c1=1024, c2=1024)
#
#     # C2PSA的输入是SPPF层的输出
#     with torch.no_grad():
#         c2psa_output = c2psa_layer(sppf_output)
#
#     print(f"Feature shape AFTER C2PSA: {c2psa_output.shape}")
#     print("\n--- 🎉🎉🎉 Full Backbone Simulation Successful! ---")


#####################################################################
# --- 模拟前向传播 ---
if __name__ == '__main__':
    # 假设您的环境中可以成功导入以下所有模块
    from ultralytics.nn.modules import Conv, C3k2, SPPF, C2PSA

    # 创建一个网络模块的列表，以模拟YAML中的顺序结构
    backbone_layers = nn.ModuleList()

    # --- 严格按照YAML和C3k2类定义来构建网络 ---

    # YAML Line 0: [-1, 1, Conv, [64, 3, 2]]
    backbone_layers.append(Conv(c1=3, c2=64, k=3, s=2))

    # YAML Line 1: [-1, 1, Conv, [128, 3, 2]]
    backbone_layers.append(Conv(c1=64, c2=128, k=3, s=2))

    # YAML Line 2: [-1, 2, C3k2, [256, False, 0.25]]
    # c1=128, c2=256, n=2, c3k=False, e=0.25
    backbone_layers.append(C3k2(c1=128, c2=256, n=2, c3k=False, e=0.25))

    # YAML Line 3: [-1, 1, Conv, [256, 3, 2]]
    backbone_layers.append(Conv(c1=256, c2=256, k=3, s=2))

    # YAML Line 4: [-1, 2, C3k2, [512, False, 0.25]]
    # c1=256, c2=512, n=2, c3k=False, e=0.25
    backbone_layers.append(C3k2(c1=256, c2=512, n=2, c3k=False, e=0.25))

    # YAML Line 5: [-1, 1, Conv, [512, 3, 2]]
    backbone_layers.append(Conv(c1=512, c2=512, k=3, s=2))

    # YAML Line 6: [-1, 2, C3k2, [512, True]]
    # c1=512, c2=512, n=2, c3k=True
    backbone_layers.append(C3k2(c1=512, c2=512, n=2, c3k=True))

    # YAML Line 7: [-1, 1, Conv, [1024, 3, 2]]
    backbone_layers.append(Conv(c1=512, c2=1024, k=3, s=2))

    # YAML Line 8: [-1, 2, C3k2, [1024, True]]
    # c1=1024, c2=1024, n=2, c3k=True
    backbone_layers.append(C3k2(c1=1024, c2=1024, n=2, c3k=True))

    # YAML Line 9: [-1, 1, SPPF, [1024, 5]]
    backbone_layers.append(SPPF(c1=1024, c2=1024, k=5))

    # YAML Line 10: [-1, 2, C2PSA, [1024]]
    # 假设 C2PSA(c1, c2, n)
    backbone_layers.append(C2PSA(c1=1024, c2=1024, n=2))
    # 创建一个虚拟输入图像
    dummy_input = torch.randn(1, 3, 640, 640)
    print(f"--- Starting Backbone Simulation ---")
    print(f"Initial Input Shape: {dummy_input.shape}\n")

    x = dummy_input

    # 逐层执行
    for i, layer in enumerate(backbone_layers):
        x = layer(x)
        print(f"Layer {i:<2} ({layer.__class__.__name__}):\t Output Shape: {x.shape}")

        # 打印出关键的P3, P4, P5输出
        if i == 3:  # Conv after C3k2
            print("  -> P3/8 Feature Map Extracted")
        if i == 5:  # Conv after C3k2
            print("  -> P4/16 Feature Map Extracted")
        if i == 7:  # Conv after C3k2
            print("  -> P5/32 Feature Map Extracted")

    print(f"\n--- Final Backbone Output Shape ---")
    print(f"After Layer 10 (C2PSA): {x.shape}")