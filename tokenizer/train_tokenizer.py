import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from tokenizers.processors import ByteLevel as ByteLevelProcessor

def train_bpe_tokenizer(files_path, output_dir, vocab_size=32000):
    print(f"Bắt đầu train BPE Tokenizer (Vocab: {vocab_size})...")
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>", "<|pad|>"],
        show_progress=True
    )
    
    # Giả lập đọc dữ liệu
    tokenizer.train(files=files_path, trainer=trainer)
    
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(out_file)
    print(f"Đã lưu tokenizer tại: {out_file}")

if __name__ == "__main__":
    # Ví dụ đường dẫn (sẽ cấu hình lại ở Milestone 4)
    print("Vui lòng cung cấp file dataset code thực tế để train.")
