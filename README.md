#### YOLO-DINOv3-ConvNext

This is a code repository for a model that replaces the Backbone of the YOLOv11 model with the DINOv3-ConvNext model ( [facebookresearch/dinov3: Reference PyTorch implementation and models for DINOv3](https://github.com/facebookresearch/dinov3) ).



##### Environment Preparation

```
conda create -n your_env_name python=3.10
conda activate your_env_name
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia # Customize according to device information
pip install ultralytics
```



##### Usability Testing

```python
python verify_model.py
```

After the usability test is passed, the training logic of the YOLO model can be reused to train on a custom dataset, where the model structure configuration file is `configs/yolo11-dinov3.yaml`.



##### Modify DINOv3 Model Size

`configs/yolo11-dinov3.yaml`

```
- [-1, 1, DINOv3ConvNextBackbone, ['dinov3-main', 'dinov3-main/checkpoints/dinov3_convnext_small.pth', 'small', True]] # 0
```

Install checkpoints from https://github.com/facebookresearch/dinov3

'dinov3-main': repo-dir

'dinov3-main/checkpoints/dinov3_convnext_small.pth': checkpoint path

'small': model size

'True': whether freeze backbone (DINOv3)