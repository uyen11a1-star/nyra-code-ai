import torch
from torch import nn

def initialize_weights(model, initializer_range=0.02):
    print("Đang khởi tạo trọng số chuẩn (Normal)...")
    for module in model.modules():
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=initializer_range)
    print("Khởi tạo hoàn tất!")
