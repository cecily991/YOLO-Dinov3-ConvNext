import torch

REPO_DIR = "dinov3-main"

# DINOv3 ConvNeXt models pretrained on web images
dinov3_convnext_tiny = torch.hub.load(
    REPO_DIR, "dinov3_convnext_tiny", source="local", weights="dinov3-main/checkpoints/dinov3_convnext_tiny.pth"
)
print(dinov3_convnext_tiny.n_storage_tokens)
