import torch
import time
import os
import sys
from torch.utils.data import DataLoader
from datasets import load_dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.config import NyraConfig
from model.transformer import NyraModel
from training.dataset import NyraDataset
from tokenizer.tokenizer import NyraTokenizer

def train_real():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = NyraConfig()
    
    # 1. Khởi tạo Model & Optimizer
    model = NyraModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    # 2. Cơ chế Resume (Nếu có checkpoint cũ thì load lại)
    checkpoint_path = "checkpoints/nyra_model_latest.pt"
    os.makedirs("checkpoints", exist_ok=True)
    start_step = 0
    if os.path.exists(checkpoint_path):
        print("Tìm thấy checkpoint, đang resume...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint['step']
        print(f"Resume từ step {start_step}")

    # 3. Tải dữ liệu bằng Streaming (Không tải về ổ cứng)
    print("Đang kết nối Streaming Dataset...")
    ds = load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)
    
    # Tokenizer
    tokenizer = NyraTokenizer("tokenizer/saved/tokenizer.json")
    
    model.train()
    step = start_step
    max_steps = 10000 # Train thử 10,000 steps
    
    start_time = time.time()
    # Batch size 8, mỗi seq 128 token
    batch_tokens = []
    
    print("Bắt đầu train...")
    for item in ds:
        # Mã hóa đoạn code thành token
        tokens = tokenizer.encode(item["content"])
        batch_tokens.extend(tokens)
        
        # Khi đủ dữ liệu cho 1 batch
        while len(batch_tokens) >= (128 * 8):
            x_batch = []
            y_batch = []
            for i in range(8):
                chunk = batch_tokens[:128]
                x_batch.append(chunk[:-1])
                y_batch.append(chunk[1:])
                batch_tokens = batch_tokens[128:] # Xóa phần đã dùng
            
            x = torch.tensor(x_batch, dtype=torch.long).to(device)
            y = torch.tensor(y_batch, dtype=torch.long).to(device)
            
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if step % 10 == 0:
                print(f"Step {step} | Loss: {loss.item():.4f} | Time: {time.time() - start_time:.2f}s")
                start_time = time.time()

            # Lưu Checkpoint mỗi 500 steps
            if step > 0 and step % 500 == 0:
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_path)
                print(f"-> Đã lưu checkpoint tại step {step}")

            step += 1
            if step >= max_steps:
                print("Đã đạt số step tối đa.")
                return

if __name__ == "__main__":
    train_real()
