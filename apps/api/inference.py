"""Suy luận cho API: nạp model một lần, tách từ, dự đoán, đánh giá tập test (có cache)."""

import sys
import threading
import unicodedata
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.data import LABEL_NAMES  # noqa: E402
from core.preprocessing import clean_text, segment_text  # noqa: E402
from model import load_model  # noqa: E402

MODEL_DIR = ROOT / "models" / "transformer-scratch-vihsd"
TEST_CSV = ROOT / "data" / "vihsd" / "vihsd" / "test.csv"

LABELS = [LABEL_NAMES[i] for i in sorted(LABEL_NAMES)]

# underthesea (CRF) không đảm bảo an toàn khi gọi song song từ nhiều thread của API.
_segment_lock = threading.Lock()


class MissingArtifactError(RuntimeError):
    pass


def check_artifacts():
    if not (MODEL_DIR / "model_state_dict.pt").exists():
        raise MissingArtifactError(
            f"Không tìm thấy checkpoint tại `{MODEL_DIR}`. Hãy chạy notebook "
            "`src/vihsd_transformer_from_scratch.ipynb` để train và lưu model trước.")
    if not TEST_CSV.exists():
        raise MissingArtifactError(
            f"Không tìm thấy tập test tại `{TEST_CSV}`. Hãy chạy notebook "
            "`src/vihsd_transformer_from_scratch.ipynb` để tải dataset ViHSD trước.")


@lru_cache(maxsize=1)
def get_model():
    model, word2idx, config = load_model(MODEL_DIR)
    segment_text("khởi động")  # nạp sẵn model tách từ của underthesea
    return model, word2idx, config


@lru_cache(maxsize=1)
def load_test_set() -> pd.DataFrame:
    return pd.read_csv(TEST_CSV)


def clear_caches():
    get_model.cache_clear()
    load_test_set.cache_clear()
    evaluate_test_set.cache_clear()


def read_csv_bytes(data) -> pd.DataFrame:
    """Đọc CSV upload; file lưu bằng Excel trên Windows thường dùng bảng mã cp1258/cp1252 thay vì UTF-8."""
    for encoding in ("utf-8-sig", "cp1258", "cp1252"):
        try:
            data.seek(0)
            df = pd.read_csv(data, encoding=encoding)
        except UnicodeDecodeError:
            continue
        # cp1258 lưu dấu tiếng Việt dạng tổ hợp (NFD) → chuẩn hoá NFC cho khớp vocab.
        for col in df.select_dtypes("object"):
            df[col] = df[col].map(lambda v: unicodedata.normalize("NFC", v) if isinstance(v, str) else v)
        return df
    raise ValueError("Không đọc được file CSV, hãy lưu lại với bảng mã UTF-8.")


def preprocess(text: str) -> tuple[str, str]:
    """Trả về (câu sau làm sạch, câu sau tách từ)."""
    cleaned = clean_text(text)
    with _segment_lock:
        return cleaned, segment_text(cleaned)


def tokenize(text: str, word2idx: dict, max_len: int):
    """Làm sạch + tách từ + mã hoá giống hệt pipeline lúc train."""
    _, segmented = preprocess(text)
    tokens = segmented.split()[:max_len]
    ids = [word2idx.get(tok, word2idx["<unk>"]) for tok in tokens]
    return tokens, ids


@torch.no_grad()
def predict_batch(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Trả về ma trận xác suất (N, 3). Câu rỗng sau làm sạch được gán NaN."""
    model, word2idx, config = get_model()
    max_len, pad_idx = config["max_len"], config["pad_idx"]

    encoded = [tokenize(t, word2idx, max_len)[1] for t in texts]
    probs = np.full((len(texts), len(LABELS)), np.nan)
    valid = [i for i, ids in enumerate(encoded) if ids]

    for start in range(0, len(valid), batch_size):
        idx = valid[start:start + batch_size]
        batch = [encoded[i] + [pad_idx] * (max_len - len(encoded[i])) for i in idx]
        logits, _ = model(torch.tensor(batch, dtype=torch.long))
        probs[idx] = torch.softmax(logits, dim=-1).numpy()
    return probs


def classification_metrics(y_true, y_pred) -> dict:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    return {
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0),
        "per_class": pd.DataFrame({"Precision": p, "Recall": r, "F1": f, "Support": s}, index=LABELS),
        "cm": confusion_matrix(y_true, y_pred, labels=LABELS),
    }


@lru_cache(maxsize=1)
def evaluate_test_set() -> dict:
    """Chạy model trên toàn bộ tập test để hiển thị số liệu thật của checkpoint hiện tại."""
    df = load_test_set()
    probs = predict_batch(df["free_text"].astype(str).tolist())
    valid = ~np.isnan(probs).any(axis=1)
    y_true = df.loc[valid, "label_id"].map(LABEL_NAMES)
    y_pred = [LABELS[i] for i in probs[valid].argmax(axis=1)]
    return classification_metrics(y_true, y_pred)


@torch.no_grad()
def analyze(text: str):
    """Dự đoán 1 câu, kèm token và attention của mọi layer (chỉ phần token thật)."""
    model, word2idx, config = get_model()
    cleaned, segmented = preprocess(text)
    tokens = segmented.split()[:config["max_len"]]
    ids = [word2idx.get(tok, word2idx["<unk>"]) for tok in tokens]
    if not ids:
        return None

    logits, attentions = model(torch.tensor([ids], dtype=torch.long))
    probs = torch.softmax(logits, dim=-1)[0].numpy()
    attn = np.stack([a[0].numpy() for a in attentions])  # (L, H, T, T)
    is_unk = [i == word2idx["<unk>"] for i in ids]
    return {"cleaned": cleaned, "segmented": segmented, "n_tokens": len(segmented.split()),
            "max_len": config["max_len"], "tokens": tokens, "is_unk": is_unk, "probs": probs, "attn": attn}
