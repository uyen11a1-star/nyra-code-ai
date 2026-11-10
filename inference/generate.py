import torch
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.config import NyraConfig
from model.transformer import NyraModel
from tokenizer.tokenizer import NyraTokenizer

# Biến toàn cục (Singleton) để load 1 lần
_model = None
_tokenizer = None
_device = None

def load_model(checkpoint_path="checkpoints/nyra_model_latest.pt"):
    global _model, _tokenizer, _device
    if _model is not None:
        return _model, _tokenizer, _device

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = NyraConfig()
    _model = NyraModel(config).to(_device)

    if os.path.exists(checkpoint_path):
        print(f"Đang tải checkpoint từ {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=_device)
        _model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Đã tải mô hình từ step {checkpoint.get('step', 'unknown')}")
    else:
        print("❌ Không tìm thấy checkpoint, dùng trọng số random.")

    # TỐI ƯU: Chạy bf16/fp16 nếu có GPU
    if _device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            _model.to(torch.bfloat16)
            print("Sử dụng bfloat16.")
        else:
            _model.to(torch.float16)
            print("Sử dụng float16.")
    
    _model.eval()
    _tokenizer = NyraTokenizer("tokenizer/saved/tokenizer.json")
    return _model, _tokenizer, _device

@torch.inference_mode()
def generate_code(prompt, max_new_tokens=100, temperature=0.7):
    model, tokenizer, device = load_model()
    
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    
    print(f"\nPrompt: {prompt}")
    print("Nyra đang viết tiếp...")
    
    past_key_values = None
    generated_ids = input_ids
    
    start_time = time.time()
    
    for _ in range(max_new_tokens):
        # Đối với bước prefill (đầu tiên), tính toán toàn bộ prompt.
        # Đối với các bước sau, chỉ tính token vừa sinh ra.
        if past_key_values is None:
            current_input_ids = input_ids
        else:
            current_input_ids = idx_next
            
        # Position IDs chuẩn xác cho RoPE
        past_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        position_ids = torch.arange(past_length, current_input_ids.shape[1] + past_length, device=device).unsqueeze(0)
        
        logits, past_key_values = model(
            current_input_ids, 
            position_ids=position_ids, 
            past_key_values=past_key_values, 
            use_cache=True
        )
        
        logits = logits[:, -1, :]
        
        # Lấy token tiếp theo (Greedy hoặc Sampling)
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            
        generated_ids = torch.cat((generated_ids, idx_next), dim=1)
        
        if idx_next.item() == tokenizer.eos_token_id:
            break
            
    end_time = time.time()
    output_text = tokenizer.decode(generated_ids[0].tolist())
    
    print("\n--- KẾT QUẢ NYRA SINH RA ---")
    print(output_text)
    print("----------------------------")
    print(f"⏱️ Tốc độ: {(max_new_tokens)/(end_time - start_time):.2f} tokens/s")

if __name__ == "__main__":
    # Test
    generate_code("def add(a, b):\n    \"\"\"Add two numbers\"\"\"", max_new_tokens=50, temperature=0.7)
    generate_code("class Dog:\n    def __init__(self, name):", max_new_tokens=50, temperature=0.7)
