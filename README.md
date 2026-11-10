# Nyra Code AI

AI coding agent with a native decoder-only Transformer trained from scratch (~300M params).

> Native model và External AI Providers được phân biệt rõ ràng. Không có pretrained weights từ LLM khác.

## Structure
- `model/` — Transformer, attention, embeddings
- `tokenizer/` — BPE tokenizer train từ đầu
- `training/` — pipeline pretraining
- `evaluation/` — perplexity, benchmarks
- `inference/` — generate, sampling
- `agent/` — multi-step coding agent
- `providers/` — provider hub (native + external)
- `sandbox/` — safe execution
- `web/` — web UI
- `notebooks/` — MoLab training notebooks

## Status
Milestone 1 — setup.
