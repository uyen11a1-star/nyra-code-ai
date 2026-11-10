import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.config import NyraConfig
from model.transformer import NyraModel
from tokenizer.tokenizer import NyraTokenizer

def generate_code(prompt, max_new_tokens=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = NyraConfig()
    model = NyraModel(config).to(device)
    
    # Load checkpoint
    checkpoint_path = "checkpoints/nyra_model_latest.pt"
    if os.path.exists(checkpoint_path):
        print("Đang tải checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Đã tải mô hình từ step {checkpoint['step']}")
    else:
        print("❌ Không tìm thấy checkpoint, dùng trọng số random.")
        
    model.eval()
    tokenizer = NyraTokenizer("tokenizer/saved/tokenizer.json")
    
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
    
    print(f"\nPrompt: {prompt}")
    print("Nyra đang viết tiếp...")
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Lấy tối đa 512 token cuối cùng làm context
            idx_cond = input_ids[:, -512:]
            
            # Tính position_ids chuẩn xác cho RoPE khi câu dài ra
            current_length = input_ids.shape[1]
            position_ids = torch.arange(current_length - idx_cond.shape[1], current_length, device=device).unsqueeze(0)
            
            logits = model(idx_cond, position_ids=position_ids)
            # Chỉ lấy token cuối cùng
            logits = logits[:, -1, :]
            
            # Greedy decoding (chọn token có xác suất cao nhất)
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            input_ids = torch.cat((input_ids, idx_next), dim=1)
            
            # Nếu gặp token kết thúc thì dừng
            if idx_next.item() == tokenizer.eos_token_id:
                break
                
    output_text = tokenizer.decode(input_ids[0].tolist())
    print("\n--- KẾT QUẢ NYRA SINH RA ---")
    print(output_text)
    print("----------------------------")

if __name__ == "__main__":
    # Thử thách Nyra với vài prompt
    prompts = [
        "def add(a, b):\n    \"\"\"Add two numbers\"\"\"",
        "class Dog:\n    def __init__(self, name):"
    ]
    for p in prompts:
        generate_code(p, max_new_tokens=50)
        print("="*40)
