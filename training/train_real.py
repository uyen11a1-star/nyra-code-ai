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
    if os.path.exists(checkpoint_path):
        print("Tìm thấy checkpoint, đang resume...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint['step']
        print(f"Resume từ step {start_step}")

    print("Đang kết nối Streaming Dataset...")
    ds = load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)
    
    # GIỮ NGUYÊN BATCH SIZE 100 NHƯ CODE CŨ ĐỂ KHÔNG BỊ TREO
    def batch_iterator(batch_size=100):
        batch = []
        for item in ds:
            batch.append(item["content"])
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
    max_steps = 10000 
    
    # TỐI ƯU AN TOÀN: Batch 64, Seq 512 (Bằng đúng code cũ nhưng gấp đôi batch)
    BATCH_SIZE = 64
    SEQ_LEN = 512
    tokens_needed = BATCH_SIZE * SEQ_LEN

    batch_tokens = []
    start_time = time.time()

    print(f"Bắt đầu train (Batch: {BATCH_SIZE}, Seq: {SEQ_LEN})...")
    
    for batch_texts in batch_iterator(batch_size=100):
        print(f"-> Đang lấy thêm 100 file code từ Hugging Face...") # In ra để bạn biết nó không bị treo
        encodings = hf_tokenizer.encode_batch(batch_texts)
        for enc in encodings:
            batch_tokens.extend(enc.ids)
            
            while len(batch_tokens) >= tokens_needed:
                chunk = torch.tensor(batch_tokens[:tokens_needed], dtype=torch.long).view(BATCH_SIZE, SEQ_LEN)
                x = chunk[:, :-1].to(device)
                y = chunk[:, 1:].to(device)
                
                del batch_tokens[:tokens_needed]
                
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
