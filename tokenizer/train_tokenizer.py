import os
import sys
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

SPECIAL_TOKENS = ["<|endoftext|>", "<|pad|>"]


def _build(vocab_size):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        # Bắt buộc: đủ 256 byte, nếu không byte chưa gặp khi train sẽ bị rơi mất lúc encode
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    return tokenizer, trainer


def _save(tokenizer, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(out_file)
    print(f"Đã lưu tokenizer tại: {out_file} (vocab thực tế: {tokenizer.get_vocab_size()})")
    return out_file


def train_bpe_tokenizer(files_path, output_dir, vocab_size=32000):
    print(f"Bắt đầu train BPE Tokenizer (Vocab: {vocab_size})...")
    tokenizer, trainer = _build(vocab_size)
    tokenizer.train(files=files_path, trainer=trainer)
    return _save(tokenizer, output_dir)


def train_bpe_from_iterator(texts, output_dir, vocab_size=32000):
    print(f"Bắt đầu train BPE Tokenizer từ iterator (Vocab: {vocab_size})...")
    tokenizer, trainer = _build(vocab_size)
    tokenizer.train_from_iterator(texts, trainer=trainer)
    out = _save(tokenizer, output_dir)
    if tokenizer.get_vocab_size() != vocab_size:
        print(f"⚠️ Vocab thực tế {tokenizer.get_vocab_size()} != {vocab_size} (dữ liệu quá ít?)")
    return out


def _stream_codeparrot(num_docs):
    from datasets import load_dataset
    ds = load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)
    ds = ds.shuffle(seed=0, buffer_size=10_000)
    for i, item in enumerate(ds):
        if i >= num_docs:
            break
        if item.get("content"):
            yield item["content"]


if __name__ == "__main__":
    # python -m tokenizer.train_tokenizer [num_docs] [vocab_size]
    num_docs = int(sys.argv[1]) if len(sys.argv) > 1 else 50_000
    vocab = int(sys.argv[2]) if len(sys.argv) > 2 else 32000
    path = train_bpe_from_iterator(_stream_codeparrot(num_docs), "tokenizer/saved", vocab_size=vocab)
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from training import hub_sync
    hub_sync.push(path, "tokenizer.json", wait=True)
