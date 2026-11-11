"""Đồng bộ checkpoint/tokenizer lên Hugging Face Hub để sống sót khi session MoLab bị kill.

Bật bằng 2 biến môi trường: HF_TOKEN (write) và NYRA_HF_REPO (vd: nguyenquochuy/nyra-code-ai-ckpt).
Không đặt -> mọi hàm no-op, train chỉ lưu local.
"""
import os


def enabled():
    return bool(os.environ.get("HF_TOKEN") and os.environ.get("NYRA_HF_REPO"))


def _api():
    from huggingface_hub import HfApi
    return HfApi(token=os.environ["HF_TOKEN"])


def pull(filename, local_dir):
    """Tải file từ repo về local_dir/filename. Trả về path hoặc None nếu không có."""
    if not enabled():
        return None
    from huggingface_hub import hf_hub_download
    try:
        return hf_hub_download(
            repo_id=os.environ["NYRA_HF_REPO"], filename=filename,
            local_dir=local_dir, token=os.environ["HF_TOKEN"],
        )
    except Exception as e:  # repo/file chưa tồn tại
        print(f"[hub] chưa có {filename} trên Hub ({type(e).__name__})")
        return None


def push(path, name, wait=False):
    """Upload nền. Trả về Future (hoặc None nếu tắt)."""
    if not enabled():
        return None
    api = _api()
    repo = os.environ["NYRA_HF_REPO"]
    api.create_repo(repo_id=repo, private=True, exist_ok=True)
    fut = api.upload_file(
        path_or_fileobj=path, path_in_repo=name, repo_id=repo,
        commit_message=f"update {name}", run_as_future=True,
    )
    if wait:
        fut.result()
    return fut
