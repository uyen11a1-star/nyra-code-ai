import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.config import NyraConfig
from model.transformer import NyraModel

def check_parameters():
    config = NyraConfig()
    model = NyraModel(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("="*40)
    print("KIỂM TRA THÔNG SỐ MÔ HÌNH NYRA NATIVE")
    print("="*40)
    print(f"Cấu hình: {config.num_hidden_layers} layers, {config.hidden_size} dims")
    print(f"Tổng Parameters : {total_params:,} ({total_params / 1_000_000:.2f}M)")
    print(f"Trainable Params: {trainable_params:,}")
    print("="*40)
    
    assert 290_000_000 < total_params < 310_000_000, "Số lượng params không nằm trong khoảng 300M!"
    print("✅ ĐẠT CHUẨN MỤC TIÊU ~300M PARAMETERS.")
    
    # Test forward pass với dữ liệu giả
    print("Đang test Forward Pass (random input)...")
    input_ids = torch.randint(0, config.vocab_size, (2, 128)) # batch=2, seq_len=128
    logits = model(input_ids)
    print(f"Shape của output logits: {logits.shape} (Kỳ vọng: [2, 128, 32000])")
    print("✅ FORWARD PASS THÀNH CÔNG.")

if __name__ == "__main__":
    check_parameters()
