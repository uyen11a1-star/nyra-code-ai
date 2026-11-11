import torch
import time
import os
import sys
import math
import queue
import signal
import threading
import numpy as np
from datasets import load_dataset
from torch.utils.data import IterableDataset, DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.config import NyraConfig
from model.transformer import NyraModel
from tokenizer.tokenizer import NyraTokenizer
from tokenizers import Tokenizer
from training import hub_sync

# ---- Cấu hình (đổi bằng biến môi trường, không cần sửa code) ----
CKPT_DIR = os.environ.get("NYRA_CKPT_DIR", "checkpoints")
SAVE_EVERY_MIN = float(os.environ.get("NYRA_SAVE_MIN", 30))          # lưu + upload mỗi N phút
SEQ_LEN = int(os.environ.get("NYRA_SEQ_LEN", 1024))
GLOBAL_BATCH = 128                                                    # số sequence / optimizer step
TARGET_TOKENS = int(os.environ.get("NYRA_TARGET_TOKENS", 5_000_000_000))
MAX_LR, MIN_LR, WARMUP_STEPS = 3e-4, 3e-5, 200
LOG_EVERY = 10
DATA_SEED = 1234


def default_micro_batch(device):
    # Ước lượng thô theo VRAM (chưa đo trên GPU thật). OOM -> đặt NYRA_BATCH nhỏ hơn.
    if device.type != "cuda":
        return 2
    gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    return 16 if gb >= 70 else 8 if gb >= 22 else 4


def prefetch(gen, size=8):
    """Chạy generator ở thread nền để tokenize/tải dữ liệu song song với GPU."""
    q = queue.Queue(maxsize=size)
    done = object()

    def work():
        try:
            for item in gen:
                q.put(item)
        except BaseException as e:
            q.put(e)
        q.put(done)

    threading.Thread(target=work, daemon=True).start()
    while True:
        item = q.get()
        if item is done:
            return
        if isinstance(item, BaseException):
            raise item
        yield item


class StreamingTokenDataset(IterableDataset):
    def __init__(self, hf_dataset, tokenizer, seq_len, eos_id, doc_batch=64):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.eos_id = eos_id
        self.doc_batch = doc_batch

    def _encode(self, texts):
        ids = []
        for enc in self.tokenizer.encode_batch(texts, add_special_tokens=False):
            ids.extend(enc.ids)
            ids.append(self.eos_id)  # ngăn cách giữa các file để model học được EOS
        return np.array(ids, dtype=np.int64)

    def _chunks(self):
        S = self.seq_len
        buf = np.empty(0, dtype=np.int64)
        texts = []
        for item in self.hf_dataset:
            content = item.get("content")
            if content:
                texts.append(content)
            if len(texts) < self.doc_batch:
                continue
            buf = np.concatenate([buf, self._encode(texts)])
            texts = []
            n = (len(buf) - 1) // S
            for k in range(n):
                yield torch.from_numpy(buf[k * S:(k + 1) * S + 1].copy())  # chunk S+1, gối 1 token
            buf = buf[n * S:]

    def __iter__(self):
        return prefetch(self._chunks())


def collate_fn(batch):
    x = torch.stack([item[:-1] for item in batch])
    y = torch.stack([item[1:] for item in batch])
    return x, y


def get_lr(step, max_steps):
    if step < WARMUP_STEPS:
        return MAX_LR * (step + 1) / WARMUP_STEPS
    progress = min(1.0, (step - WARMUP_STEPS) / max(1, max_steps - WARMUP_STEPS))
    return MIN_LR + 0.5 * (MAX_LR - MIN_LR) * (1 + math.cos(math.pi * progress))


def save_checkpoint(path, payload):
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)  # atomic: crash giữa chừng không làm hỏng checkpoint cũ


def train_real():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    print(f"Thiết bị: {device.type.upper()}")
    if use_cuda:
        p = torch.cuda.get_device_properties(0)
        print(f"GPU: {p.name} | VRAM: {p.total_memory / 2**30:.1f} GB")
    # bf16 chỉ native từ Ampere (SM80) trở lên; GPU cũ (T4...) dùng fp16 + GradScaler
    amp_dtype = torch.bfloat16 if use_cuda and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_cuda and amp_dtype == torch.float16))
    print(f"Mixed precision: {amp_dtype if use_cuda else 'tắt (CPU)'}")

    config = NyraConfig()
    model = NyraModel(config).to(device)
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
        lr=MAX_LR, betas=(0.9, 0.95), fused=use_cuda,
    )

    tokenizer_path = "tokenizer/saved/tokenizer.json"
    if not os.path.exists(tokenizer_path):
        hub_sync.pull("tokenizer.json", "tokenizer/saved")
    if not os.path.exists(tokenizer_path):
        raise SystemExit("Chưa có tokenizer. Chạy trước: python -m tokenizer.train_tokenizer")
    tokenizer = NyraTokenizer(tokenizer_path)
    hf_tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab = hf_tokenizer.get_vocab_size()
    if vocab > config.vocab_size or vocab < 0.9 * config.vocab_size:
        raise SystemExit(f"Tokenizer vocab={vocab} không khớp config.vocab_size={config.vocab_size}. Train lại tokenizer.")

    ckpt_path = os.path.join(CKPT_DIR, "nyra_model_latest.pt")
    os.makedirs(CKPT_DIR, exist_ok=True)
    if not os.path.exists(ckpt_path):
        hub_sync.pull("nyra_model_latest.pt", CKPT_DIR)
    start_step, total_tokens_trained = 0, 0
    if os.path.exists(ckpt_path):
        print("Tìm thấy checkpoint, đang resume...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt["step"]
        total_tokens_trained = ckpt.get("total_tokens", 0)
        del ckpt
        print(f"Resume từ step {start_step} (Đã học {total_tokens_trained:,} tokens)")

    print("Đang kết nối Streaming Dataset...")
    ds = load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)
    # seed đổi theo step để mỗi lần resume không lặp lại đúng phần đầu dataset
    ds = ds.shuffle(seed=DATA_SEED + start_step, buffer_size=10_000)

    BATCH_SIZE = int(os.environ.get("NYRA_BATCH", default_micro_batch(device)))
    GRAD_ACCUM = max(1, GLOBAL_BATCH // BATCH_SIZE)
    tokens_per_step = BATCH_SIZE * SEQ_LEN * GRAD_ACCUM
    max_steps = TARGET_TOKENS // tokens_per_step

    dataset = StreamingTokenDataset(ds, hf_tokenizer, SEQ_LEN, tokenizer.eos_token_id)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0,
                            collate_fn=collate_fn, drop_last=True, pin_memory=use_cuda)

    print(f"Mục tiêu: {TARGET_TOKENS:,} tokens | Micro-batch: {BATCH_SIZE} x accum {GRAD_ACCUM} | Seq: {SEQ_LEN}")
    print(f"Tokens/step: {tokens_per_step:,} | Tổng số step: {max_steps:,}")

    state = {"step": start_step, "tokens": total_tokens_trained, "pending": None, "last_save": time.time()}

    def save(final=False):
        save_checkpoint(ckpt_path, {
            "step": state["step"],
            "total_tokens": state["tokens"],
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        })
        print(f"-> Đã lưu checkpoint tại step {state['step']}")
        pending = state["pending"]
        if pending is not None and pending.done() and pending.exception():
            print(f"⚠️ Upload Hub lần trước lỗi: {pending.exception()}")
        if final and pending is not None:
            pending.result()
            pending = None
        if pending is None or pending.done():
            state["pending"] = hub_sync.push(ckpt_path, "nyra_model_latest.pt", wait=final)
        state["last_save"] = time.time()

    def on_term(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, on_term)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    micro = 0
    loss_sum = torch.zeros((), device=device)
    last_log_time, last_log_step = time.time(), state["step"]

    try:
        for x, y in dataloader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_cuda):
                logits, _ = model(x)
                loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            scaler.scale(loss / GRAD_ACCUM).backward()
            loss_sum += loss.detach() / GRAD_ACCUM
            micro += 1
            if micro % GRAD_ACCUM:
                continue

            lr = get_lr(state["step"], max_steps)
            for g in optimizer.param_groups:
                g["lr"] = lr
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            state["step"] += 1
            state["tokens"] += tokens_per_step
            step_loss = loss_sum.item()
            loss_sum.zero_()

            if state["step"] % LOG_EVERY == 0:
                if not math.isfinite(step_loss):
                    raise RuntimeError(f"Loss = {step_loss} tại step {state['step']} (phân kỳ). Dừng, không lưu.")
                now = time.time()
                tps = tokens_per_step * (state["step"] - last_log_step) / max(now - last_log_time, 1e-9)
                mem = f" | VRAM peak: {torch.cuda.max_memory_allocated() / 2**30:.1f} GB" if use_cuda else ""
                print(f"Step {state['step']} | Loss: {step_loss:.4f} | LR: {lr:.2e} | "
                      f"Tokens: {state['tokens']:,} / {TARGET_TOKENS:,} | Speed: {tps:.0f} tok/s{mem}")
                last_log_time, last_log_step = now, state["step"]

            if time.time() - state["last_save"] >= SAVE_EVERY_MIN * 60:
                save()

            if state["step"] >= max_steps:
                print(f"🎉 ĐÃ ĐẠT MỤC TIÊU {TARGET_TOKENS:,} TOKENS!")
                break
    except KeyboardInterrupt:
        print("Nhận tín hiệu dừng, đang lưu checkpoint...")
    save(final=True)


if __name__ == "__main__":
    train_real()
