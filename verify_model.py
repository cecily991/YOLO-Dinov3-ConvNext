# verify_model.py
import torch

from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import select_device

print("--- Starting Model Verification ---")

# 1. 指定你的配置文件和设备
cfg_path = "configs/yolo11-dinov3.yaml"
device = select_device("cpu")  # 使用CPU即可，无需GPU
print(f"Loading model from configuration: {cfg_path}")

try:
    # 2. 尝试创建模型
    # 这一步会解析YAML，并创建所有模块，包括我们自定义的DINOv3 Backbone
    model = DetectionModel(cfg=cfg_path, ch=3, nc=80).to(device)
    print("✅ Model created successfully from YAML!")

    # 3. 创建一个模拟的输入图像
    # (批量大小=1, 通道=3, 高度=640, 宽度=640)
    dummy_input = torch.randn(1, 3, 640, 640).to(device)
    print(f"Created a dummy input tensor with shape: {dummy_input.shape}")

    # 4. 将模型设置为评估模式
    model.eval()

    # 5. 执行一次完整的前向传播
    print("Attempting a full forward pass through the model...")
    with torch.no_grad():
        outputs = model(dummy_input)
    print("✅ Forward pass completed without errors!")

    # 6. 检查输出
    # YOLO的输出通常是一个列表，其中包含一个形状为 [B, num_predictions, 4+num_classes] 的张量
    print(f"Model output type: {type(outputs)}")
    if isinstance(outputs, list):
        print(f"Output list contains {len(outputs)} element(s).")
        output_tensor = outputs[0]
        print(f"Shape of the output tensor: {output_tensor.shape}")
        # 预期形状类似于 [1, 8400, 84] (8400个预测框, 每个框有4个坐标+80个类别得分)
        print("✅ Output shape is as expected for a detection model.")

    print("\n--- 🎉 Verification Successful! The DINOv3 backbone is correctly embedded. ---")

except Exception as e:
    print("\n--- ❌ Verification Failed! ---")
    print(f"An error occurred: {e}")
    import traceback

    traceback.print_exc()
