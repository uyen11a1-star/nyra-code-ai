import torch
import time
from torch.utils.data import DataLoader
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import NyraConfig
from model.transformer import NyraModel
from training.dataset import NyraDataset

def train_smoke_test():
    print("=== BẮT ĐẦU QUÁ TRÌNH TRAIN (SMOKE TEST) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị đang sử dụng: {device.type.upper()}")
    if device.type == 'cuda':
        print(f"Tên GPU: {torch.cuda.get_device_name(0)}")

    config = NyraConfig()
    model = NyraModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

    print("Đang tạo dữ liệu giả lập...")
    dummy_token_ids = torch.randint(0, config.vocab_size, (100000,)).tolist()
    dataset = NyraDataset(dummy_token_ids, seq_len=128)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True) # Tăng batch size vì có GPU

    # Tự động bật AMP (Mixed Precision) nếu có GPU
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    model.train()
    step = 0
    max_steps = 3 # Chạy thử 3 bước
    
    start_time = time.time()
    for x, y in dataloader:
        if step >= max_steps:
            break

        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        # Forward pass với Mixed Precision
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        print(f"Step {step+1}/{max_steps} | Loss: {loss.item():.4f} | Time: {time.time() - start_time:.2f}s")
        step += 1

    print("✅ TRAINING LOOP HOẠT ĐỘNG TỐT (Forward + Backward + Update OK).")

if __name__ == "__main__":
    train_smoke_test()
