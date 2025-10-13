from module import *  # import class Model từ module của bạn
import torch

def load(numClasses, device):
    in_channels = 1
    base_channels = 64
    kernel_size = 3
    padding = kernel_size // 2
    max_depth = 6
    skip = 2
    num_blocks = 4

    model = Model(
        in_channels=in_channels,
        numClasses=numClasses,
        base_channels=base_channels,
        kernel_size=kernel_size,
        padding=padding,
        max_depth=max_depth,
        skip=skip,
        num_blocks=num_blocks
    )
    # model = Backbone(numClasses)
    model = model.to(device, memory_format=torch.channels_last)
    # Đếm số lượng tham số trainable
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model.__class__.__name__}")
    print(f"Số tham số huấn luyện: {num_params:,}")

    return model
