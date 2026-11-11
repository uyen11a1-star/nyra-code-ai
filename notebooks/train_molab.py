# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "torch",
#     "numpy",
#     "tokenizers>=0.15.0",
#     "datasets>=2.16.0",
#     "huggingface_hub",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    import signal
    import subprocess
    import sys
    from pathlib import Path

    REPO_URL = "https://github.com/uyen11a1-star/nyra-code-ai.git"
    REPO_DIR = Path("nyra-code-ai").resolve()
    LOG = Path("train.log").resolve()
    PIDFILE = Path("train.pid").resolve()

    def alive():
        try:
            os.kill(int(PIDFILE.read_text()), 0)
            return True
        except Exception:
            return False

    return (
        LOG,
        PIDFILE,
        REPO_DIR,
        REPO_URL,
        alive,
        mo,
        os,
        signal,
        subprocess,
        sys,
    )


@app.cell
def _(mo):
    hf_token = mo.ui.text(label="HF token (quyền write)", kind="password", full_width=True)
    hf_repo = mo.ui.text(label="HF repo (vd: nguyenquochuy/nyra-code-ai-ckpt)", full_width=True)
    btn_setup = mo.ui.run_button(label="1) Clone / pull repo")
    btn_tok = mo.ui.run_button(label="2) Train tokenizer (làm 1 lần)")
    btn_train = mo.ui.run_button(label="3) Train / Resume", kind="success")
    btn_stop = mo.ui.run_button(label="Dừng (lưu checkpoint)", kind="danger")
    mo.vstack([hf_token, hf_repo, mo.hstack([btn_setup, btn_tok, btn_train, btn_stop])])
    return btn_setup, btn_stop, btn_tok, btn_train, hf_repo, hf_token


@app.cell
def _(hf_repo, hf_token, os):
    env = {
        **os.environ,
        "HF_TOKEN": hf_token.value.strip(),
        "NYRA_HF_REPO": hf_repo.value.strip(),
        "PYTHONUNBUFFERED": "1",
    }
    return (env,)


@app.cell
def _(REPO_DIR, REPO_URL, btn_setup, mo, subprocess):
    mo.stop(not btn_setup.value)
    if (REPO_DIR / ".git").exists():
        _cmd = ["git", "-C", str(REPO_DIR), "pull", "--ff-only"]
    else:
        _cmd = ["git", "clone", REPO_URL, str(REPO_DIR)]
    _r = subprocess.run(_cmd, capture_output=True, text=True)
    mo.md(f"```\n{_r.stdout}{_r.stderr}\n```")
    return


@app.cell
def _(REPO_DIR, btn_tok, env, mo, subprocess, sys):
    mo.stop(not btn_tok.value)
    mo.stop(
        not (env["HF_TOKEN"] and env["NYRA_HF_REPO"]),
        mo.md("⚠️ Điền HF token + HF repo trước (tokenizer phải lưu lên Hub, nếu không mất khi session bị kill)."),
    )
    with mo.status.spinner(title="Đang train tokenizer..."):
        _r = subprocess.run(
            [sys.executable, "-m", "tokenizer.train_tokenizer"],
            cwd=REPO_DIR, env=env, capture_output=True, text=True,
        )
    mo.md(f"```\n{(_r.stdout + _r.stderr)[-2500:]}\n```")
    return


@app.cell
def _(LOG, PIDFILE, REPO_DIR, alive, btn_train, env, mo, subprocess, sys):
    mo.stop(not btn_train.value)
    mo.stop(
        not (env["HF_TOKEN"] and env["NYRA_HF_REPO"]),
        mo.md("⚠️ Điền HF token + HF repo trước (checkpoint phải lưu lên Hub)."),
    )
    mo.stop(alive(), mo.md("Đang có tiến trình train chạy rồi."))
    _log = open(LOG, "a")
    _p = subprocess.Popen(
        [sys.executable, "-u", "-m", "training.train_real"],
        cwd=REPO_DIR, env=env, stdout=_log, stderr=subprocess.STDOUT, start_new_session=True,
    )
    PIDFILE.write_text(str(_p.pid))
    mo.md(f"Đã chạy train (pid {_p.pid}). Log bên dưới.")
    return


@app.cell
def _(PIDFILE, alive, btn_stop, mo, os, signal):
    mo.stop(not btn_stop.value)
    if alive():
        os.kill(int(PIDFILE.read_text()), signal.SIGTERM)
        _msg = "Đã gửi SIGTERM, train sẽ lưu checkpoint rồi thoát."
    else:
        _msg = "Không có tiến trình train nào đang chạy."
    mo.md(_msg)
    return


@app.cell
def _(mo):
    refresh = mo.ui.refresh(options=["5s", "10s", "30s"], default_interval="10s", label="Tự làm mới log")
    refresh
    return (refresh,)


@app.cell
def _(LOG, alive, mo, refresh):
    refresh.value
    _tail = "\n".join(LOG.read_text(errors="replace").splitlines()[-25:]) if LOG.exists() else "(chưa có log)"
    mo.md(f"**Trạng thái:** {'🟢 đang chạy' if alive() else '⚪ không chạy'}\n\n```\n{_tail}\n```")
    return


if __name__ == "__main__":
    app.run()
