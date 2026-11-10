import torch
import time
import os
import sys
from datasets import load_dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.config import NyraConfig
from model.transformer import NyraModel
from tokenizer.tokenizer import NyraTokenizer

def train_real():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị: {device.type.upper()}")
    config = NyraConfig()
    
    model = NyraModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    # Load Tokenizer để lấy thông số
    tokenizer = NyraTokenizer("tokenizer/saved/tokenizer.json")

    # Cơ chế Resume
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

    print("Đang kết nối Streaming Dataset...")
    ds = load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)
    
    # TỐI ƯU HÓA: Gom 100 file code lại với nhau để tokenize 1 lần (nhanh hơn rất nhiều)
    def batch_iterator(batch_size=100):
        batch = []
        for item in ds:
            batch.append(item["content"])
            if len(batch) == batch_size:
                yield batch
                batch = []

    print("Bắt đầu tokenize dữ liệu (Đa luồng)...")
    # Sử dụng tokenizer của HuggingFace để tokenize song song cực nhanh
    from tokenizers import Tokenizer
    hf_tokenizer = Tokenizer.from_file("tokenizer/saved/tokenizer.json")
    
    # Đưa luồng dữ liệu qua tokenizer
    tokenized_ds = hf_tokenizer.encode_batch(batch_iterator()) 

    model.train()
    step = start_step
    max_steps = 10000 # Train 10,000 steps
    
    # Tăng kích thước Batch và Seq lên tối đa để tận dụng 102GB VRAM
    BATCH_SIZE = 32
    SEQ_LEN = 512
    tokens_needed = BATCH_SIZE * SEQ_LEN

    batch_tokens = []
    start_time = time.time()

    print(f"Bắt đầu train (Batch: {BATCH_SIZE}, Seq: {SEQ_LEN})...")
    for enc in tokenized_ds:
        batch_tokens.extend(enc.ids)
        
        while len(batch_tokens) >= tokens_needed:
            x_batch, y_batch = [], []
            for i in range(BATCH_SIZE):
                chunk = batch_tokens[:SEQ_LEN]
                x_batch.append(chunk[:-1])
                y_batch.append(chunk[1:])
                batch_tokens = batch_tokens[SEQ_LEN:]
            
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
                print(f"Step {step} | Loss: {loss.item():.4f} | Time: {time.time() - start_time:.2f}s | Tokens/s: {tokens_needed/(time.time() - start_time):.0f}")
                start_time = time.time()

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
