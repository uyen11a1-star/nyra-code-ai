"""Trộn dữ liệu nhiều ngôn ngữ từ The Stack (dedup, license permissive) + tách validation theo hash.

Cần: tài khoản HF đã bấm 'Access repository' (đồng ý điều khoản) ở trang dataset, và HF_TOKEN của tài khoản đó.
Kiểm tra nhanh: python -m training.data_mix
"""
import hashlib
import os

DATASET = "bigcode/the-stack-dedup"
# tên thư mục data/<tên> -> trọng số (%). Thiếu thư mục nào thì bỏ và chuẩn hóa lại (có in cảnh báo).
LANG_MIX = {
    "python": 30, "javascript": 15, "typescript": 12, "java": 10, "cpp": 8, "c": 6,
    "html": 4, "css": 2, "sql": 3, "shell": 3, "markdown": 7,
}
VAL_MOD = 200  # ~0.5% tài liệu (theo hash nội dung) dành cho validation, không bao giờ dùng để train


def is_val(text):
    h = hashlib.md5(text.encode("utf-8", "ignore")).hexdigest()
    return int(h[:8], 16) % VAL_MOD == 0


def available_langs():
    from huggingface_hub import HfApi
    try:
        entries = list(HfApi(token=os.environ.get("HF_TOKEN")).list_repo_tree(
            DATASET, path_in_repo="data", repo_type="dataset"))
    except Exception as e:
        raise SystemExit(
            f"Không đọc được {DATASET} ({type(e).__name__}). Đăng nhập đúng tài khoản HF, mở "
            f"https://huggingface.co/datasets/{DATASET} và bấm 'Access repository' để đồng ý điều khoản.")
    names = {e.path.split("/")[-1] for e in entries}
    present = [l for l in LANG_MIX if l in names]
    missing = [l for l in LANG_MIX if l not in names]
    if missing:
        print(f"⚠️ Không có thư mục data/ cho: {missing} -> bỏ qua, chuẩn hóa lại trọng số")
    if not present:
        raise SystemExit(f"Không khớp thư mục nào. Thư mục có sẵn (mẫu): {sorted(names)[:40]}")
    return present


def _load_lang(lang):
    from datasets import load_dataset
    return load_dataset(DATASET, data_dir=f"data/{lang}", split="train", streaming=True,
                        token=os.environ.get("HF_TOKEN"))


def make_stream(seed, want_val, loader=None, langs=None):
    """Stream đã trộn các ngôn ngữ theo LANG_MIX. want_val=True -> chỉ tài liệu validation."""
    from datasets import interleave_datasets
    loader = loader or _load_lang
    langs = langs or available_langs()
    dss, weights = [], []
    for lang in langs:
        d = loader(lang)
        d = d.filter(lambda t: is_val(t) == want_val, input_columns="content")
        d = d.select_columns(["content"])
        dss.append(d.shuffle(seed=seed, buffer_size=2000))
        weights.append(LANG_MIX[lang])
    probs = [w / sum(weights) for w in weights]
    return interleave_datasets(dss, probabilities=probs, seed=seed, stopping_strategy="all_exhausted")


if __name__ == "__main__":
    langs = available_langs()
    print("Ngôn ngữ dùng được:", langs)
    for lang in langs:
        ex = next(iter(_load_lang(lang)))
        print(f"{lang}: OK (mẫu đầu {len(ex['content'])} ký tự)")
