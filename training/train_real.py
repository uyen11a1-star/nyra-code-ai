import torch
import time
import os
import sys
from datasets import load_dataset, interleave_datasets

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

    tokenizer_path = "tokenizer/saved/tokenizer.json"
    if not os.path.exists(tokenizer_path):
        print("Chưa tìm thấy Tokenizer, đang tự động tạo mới...")
        os.makedirs("data", exist_ok=True)
        with open("data/sample.py", "w") as f:
            f.write("def test_fn():\n    return 'Hello Nyra'\n")
        from tokenizer.train_tokenizer import train_bpe_tokenizer
        train_bpe_tokenizer(["data/sample.py"], "tokenizer/saved", vocab_size=32000)

    tokenizer = NyraTokenizer(tokenizer_path)

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

    print("Đang kết nối Streaming Datasets...")
    ds_code = load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)
    ds_wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    
    ds = interleave_datasets([ds_code, ds_wiki], probabilities=[0.8, 0.2], seed=42)
    
    def batch_iterator(batch_size=200):
        batch = []
        for item in ds:
            # SỬA LỖI TẠI ĐÂY: Lấy dữ liệu an toàn, bỏ qua nếu rỗng
            if "content" in item and item["content"] is not None:
                text = item["content"]
            elif "text" in item and item["text"] is not None:
                text = item["text"]
            else:
                continue # Bỏ qua nếu không có dữ liệu
            
            batch.append(str(text)) # Ép kiểu chắc chắn thành string
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    print("Bắt đầu tokenize dữ liệu...")
    from tokenizers import Tokenizer
    hf_tokenizer = Tokenizer.from_file(tokenizer_path)

    model.train()
    step = start_step
    TARGET_TOKENS = 5_000_000_000
    BATCH_SIZE = 32
    SEQ_LEN = 1024
    tokens_per_step = BATCH_SIZE * SEQ_LEN
    max_steps = TARGET_TOKENS // tokens_per_step
    
    batch_tokens = []
    start_time = time.time()

    print(f"Mục tiêu: {TARGET_TOKENS:,} tokens | Batch: {BATCH_SIZE} | Seq: {SEQ_LEN}")
    print(f"Tổng số step cần: {max_steps:,} (Đã chạy: {step:,})")
    
    for batch_texts in batch_iterator(batch_size=200):
        # Lọc lại lần nữa cho chắc chắn 100% không có phần tử None
        batch_texts = [t for t in batch_texts if t]
        if not batch_texts:
            continue
            
        encodings = hf_tokenizer.encode_batch(batch_texts)
        for enc in encodings:
            batch_tokens.extend(enc.ids)
            
            while len(batch_tokens) >= tokens_per_step:
                chunk = torch.tensor(batch_tokens[:tokens_per_step], dtype=torch.long).view(BATCH_SIZE, SEQ_LEN)
                x = chunk[:, :-1].to(device)
                y = chunk[:, 1:].to(device)
                
                del batch_tokens[:tokens_per_step]
                
                optimizer.zero_grad()
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
                    logits = model(x)
                    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                total_tokens_trained += tokens_per_step
                
                if step % 10 == 0:
                    elapsed = time.time() - start_time
                    tokens_per_sec = tokens_per_step / elapsed if elapsed > 0 else 0
                    print(f"Step {step} | Loss: {loss.item():.4f} | Tokens: {total_tokens_trained:,}/{TARGET_TOKENS:,} | Speed: {tokens_per_sec:.0f} tok/s")
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
