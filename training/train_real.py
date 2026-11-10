import torch
import time
import os
import sys
from datasets import load_dataset
from torch.utils.data import IterableDataset, DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.config import NyraConfig
from model.transformer import NyraModel
from tokenizer.tokenizer import NyraTokenizer
from tokenizers import Tokenizer

class StreamingTokenDataset(IterableDataset):
    def __init__(self, hf_dataset, tokenizer, seq_len):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __iter__(self):
        buffer = []
        for item in self.hf_dataset:
            if "content" in item and item["content"] is not None:
                tokens = self.tokenizer.encode(item["content"]).ids
                buffer.extend(tokens)
                while len(buffer) >= self.seq_len + 1:
                    chunk = buffer[:self.seq_len + 1]
                    yield torch.tensor(chunk, dtype=torch.long)
                    buffer = buffer[self.seq_len:] 

def collate_fn(batch):
    x = torch.stack([item[:-1] for item in batch])
    y = torch.stack([item[1:] for item in batch])
    return x, y

def train_real():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị: {device.type.upper()}")
    config = NyraConfig()
    
    model = NyraModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    tokenizer_path = "tokenizer/saved/tokenizer.json"
    if not os.path.exists(tokenizer_path):
        print("Chưa tìm thấy Tokenizer, đang tự động tạo mới...")
        os.makedirs("data", exist_ok=True)
        with open("data/sample.py", "w") as f:
            f.write("def test_fn():\n    return 'Hello Nyra'\n")
        from tokenizer.train_tokenizer import train_bpe_tokenizer
        train_bpe_tokenizer(["data/sample.py"], "tokenizer/saved", vocab_size=32000)

    tokenizer = NyraTokenizer(tokenizer_path)
    hf_tokenizer = Tokenizer.from_file(tokenizer_path)

    checkpoint_path = "checkpoints/nyra_model_latest.pt"
    os.makedirs("checkpoints", exist_ok=True)
    start_step = 0
    total_tokens_trained = 0
    if os.path.exists(checkpoint_path):
        print("Tìm thấy checkpoint, đang resume...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint['step']
        total_tokens_trained = checkpoint.get('total_tokens', 0)
        print(f"Resume từ step {start_step} (Đã học {total_tokens_trained:,} tokens)")

    print("Đang kết nối Streaming Dataset...")
    ds = load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)
    
    # CẤU HÌNH CHỐNG OOM (Gradient Accumulation)
    BATCH_SIZE = 16          # Giảm batch size thực tế trên VRAM
    SEQ_LEN = 1024           # Giữ seq len vừa phải
    GRAD_ACCUM = 8           # Tích lũy 8 bước -> Tương đương Batch Size = 128
    
    NUM_WORKERS = 0          # Đọc dữ liệu đa luồng
    
    dataset = StreamingTokenDataset(ds, hf_tokenizer, SEQ_LEN)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS, 
        collate_fn=collate_fn,
        prefetch_factor=2
    )

    model.train()
    step = start_step
    TARGET_TOKENS = 5_000_000_000
    
    # Tổng token thực tế học được trong 1 vòng lặp (step)
    tokens_per_step = BATCH_SIZE * SEQ_LEN * GRAD_ACCUM
    max_steps = TARGET_TOKENS // tokens_per_step
    
    start_time = time.time()
    print(f"Mục tiêu: {TARGET_TOKENS:,} tokens | Batch: {BATCH_SIZE}x{GRAD_ACCUM} | Seq: {SEQ_LEN}")
    print(f"Tokens/step: {tokens_per_step:,} | Tổng số step: {max_steps:,}")

    optimizer.zero_grad()
    
    for i, (x, y) in enumerate(dataloader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            # Chia loss cho số bước tích lũy
            loss = loss / GRAD_ACCUM 

        scaler.scale(loss).backward()

        # Chỉ cập nhật trọng số khi đã tích lũy đủ
        if (i + 1) % GRAD_ACCUM == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            total_tokens_trained += tokens_per_step
            
            if step % 10 == 0:
                elapsed = time.time() - start_time
                tokens_per_sec = tokens_per_step / elapsed if elapsed > 0 else 0
                # Nhân lại loss thực tế để in ra
                print(f"Step {step} | Loss: {loss.item() * GRAD_ACCUM:.4f} | Tokens: {total_tokens_trained:,} / {TARGET_TOKENS:,} | Speed: {tokens_per_sec:.0f} tok/s")
                start_time = time.time()

            if step > 0 and step % 500 == 0:
                torch.save({
                    'step': step,
                    'total_tokens': total_tokens_trained,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_path)
                print(f"-> Đã lưu checkpoint tại step {step}")

            step += 1
            if step >= max_steps:
                print(f"🎉 ĐÃ ĐẠT MỤC TIÊU {TARGET_TOKENS:,} TOKENS!")
                return

if __name__ == "__main__":
    train_real()
