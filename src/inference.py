"""Nạp checkpoint và suy luận trực tiếp bằng PyTorch (không qua API).

Tiền xử lý dùng chung `src/core/preprocessing.py` với notebook để mã hoá câu giống hệt lúc train.
"""

import threading
from functools import lru_cache
from pathlib import Path

import torch

from .core.preprocessing import clean_text, segment_text
from .model import load_model

ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT / "models"
ARTIFACTS = ("model_state_dict.pt", "vocab.json", "config.json")

# underthesea (CRF) không đảm bảo an toàn khi gọi song song từ nhiều thread.
_segment_lock = threading.Lock()


class MissingArtifactError(RuntimeError):
    pass


def check_artifacts():
    missing = [name for name in ARTIFACTS if not (MODEL_DIR / name).exists()]
    if missing:
        raise MissingArtifactError(
            f"Không tìm thấy {', '.join(missing)} trong `{MODEL_DIR}`. Hãy chạy notebook "
            "`notebook/vihsd_transformer_from_scratch.ipynb` để train và lưu model trước.")


@lru_cache(maxsize=1)
def get_model():
    model, word2idx, config = load_model(MODEL_DIR)
    segment_text("khởi động")  # nạp sẵn model tách từ của underthesea
    return model, word2idx, config


def preprocess(text: str) -> tuple[str, str]:
    """Trả về (câu sau làm sạch, câu sau tách từ)."""
    cleaned = clean_text(text)
    with _segment_lock:
        return cleaned, segment_text(cleaned)


@torch.no_grad()
def analyze(text: str):
    """Dự đoán 1 câu, kèm token và attention của mọi layer (chỉ phần token thật).

    Trả về None nếu câu rỗng sau khi làm sạch.
    """
    model, word2idx, config = get_model()
    cleaned, segmented = preprocess(text)
    tokens = segmented.split()[:config["max_len"]]
    ids = [word2idx.get(tok, word2idx["<unk>"]) for tok in tokens]
    if not ids:
        return None

    logits, attentions = model(torch.tensor([ids], dtype=torch.long))
    probs = torch.softmax(logits, dim=-1)[0].numpy()
    attn = torch.stack([a[0] for a in attentions]).numpy()  # (L, H, T, T)
    is_unk = [i == word2idx["<unk>"] for i in ids]
    return {"cleaned": cleaned, "segmented": segmented, "n_tokens": len(segmented.split()),
            "max_len": config["max_len"], "tokens": tokens, "is_unk": is_unk, "probs": probs, "attn": attn}
