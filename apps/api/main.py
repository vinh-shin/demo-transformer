"""API FastAPI cho demo phát hiện bình luận độc hại tiếng Việt.

Chạy từ thư mục gốc project:
    make start      build giao diện rồi chạy API, mở http://localhost:8000
    make api        chỉ chạy API (uvicorn main:app --app-dir apps/api --port 8000)

Nếu đã build frontend (apps/web/dist), API phục vụ luôn giao diện tại cùng cổng.
"""

import io
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

sys.path.insert(0, str(Path(__file__).resolve().parent))

import inference  # noqa: E402
from inference import LABEL_NAMES, LABELS  # noqa: E402

WEB_DIST = inference.ROOT / "apps" / "web" / "dist"
MAX_TEXT_CHARS = 2_000
MAX_BATCH_ROWS = 10_000
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

_state = {"error": None}
_metrics_lock = threading.Lock()


def _warm_up():
    try:
        inference.check_artifacts()
        inference.get_model()
        _test_metrics()  # tập test mất vài chục giây, tính sẵn để giao diện không phải chờ
    except Exception as e:  # noqa: BLE001 - lỗi được trả về qua /api/health
        _state["error"] = str(e)


@asynccontextmanager
async def lifespan(_: FastAPI):
    threading.Thread(target=_warm_up, daemon=True).start()
    yield


app = FastAPI(title="ViHSD Transformer API", version="1.0.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_ready():
    try:
        inference.check_artifacts()
    except inference.MissingArtifactError as e:
        raise HTTPException(503, str(e)) from e


def _test_metrics() -> dict:
    with _metrics_lock:
        return inference.evaluate_test_set()


def _metrics_json(m: dict) -> dict:
    return {
        "n": int(m["n"]),
        "accuracy": float(m["accuracy"]),
        "macro_f1": float(m["macro_f1"]),
        "per_class": [
            {"label": label, "precision": float(row["Precision"]), "recall": float(row["Recall"]),
             "f1": float(row["F1"]), "support": int(row["Support"])}
            for label, row in m["per_class"].iterrows()
        ],
        "confusion_matrix": m["cm"].tolist(),
    }


def _normalize_labels(values: pd.Series) -> list[str | None]:
    """Nhận nhãn dạng 0/1/2 hoặc CLEAN/OFFENSIVE/HATE; giá trị khác thành None."""
    def norm(v):
        if pd.isna(v):
            return None
        if isinstance(v, str):
            v = v.strip()
            if v.upper() in LABELS:
                return v.upper()
            if v.isdigit():
                v = int(v)
        try:
            return LABEL_NAMES.get(int(v))
        except (TypeError, ValueError):
            return None
    return [norm(v) for v in values]


def _classify(texts: list[str], truth: pd.Series | None) -> tuple[list[dict], dict]:
    if not texts:
        raise HTTPException(400, "Không có bình luận nào để phân loại.")
    if len(texts) > MAX_BATCH_ROWS:
        raise HTTPException(413, f"Tối đa {MAX_BATCH_ROWS:,} bình luận mỗi lần.")

    probs = inference.predict_batch(texts)
    valid = ~np.isnan(probs).any(axis=1)
    preds = [LABELS[int(np.argmax(p))] if ok else None for p, ok in zip(probs, valid)]

    rows = [
        {"prediction": pred, "probs": p.round(4).tolist() if ok else None}
        for pred, p, ok in zip(preds, probs, valid)
    ]
    counts = {label: preds.count(label) for label in LABELS}
    summary = {"total": len(texts), "empty": int((~valid).sum()), "counts": counts, "metrics": None}

    if truth is not None:
        truth = _normalize_labels(truth)
        for row, t in zip(rows, truth):
            row["true_label"] = t
            row["correct"] = None if t is None or row["prediction"] is None else t == row["prediction"]
        mask = [t is not None and p is not None for t, p in zip(truth, preds)]
        y_true = [t for t, m in zip(truth, mask) if m]
        y_pred = [p for p, m in zip(preds, mask) if m]
        if y_true:
            summary["metrics"] = _metrics_json(inference.classification_metrics(y_true, y_pred))
    return rows, summary


def _records(df: pd.DataFrame) -> list[dict]:
    """DataFrame → list dict an toàn cho JSON (NaN thành None, kiểu numpy thành kiểu Python)."""
    def clean(v):
        if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA:
            return None
        return v.item() if isinstance(v, np.generic) else v
    return [{k: clean(v) for k, v in rec.items()} for rec in df.to_dict(orient="records")]


async def _read_upload(file: UploadFile) -> pd.DataFrame:
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File CSV tối đa 10 MB.")
    try:
        df = inference.read_csv_bytes(io.BytesIO(data))
    except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        raise HTTPException(400, f"Không đọc được file CSV: {e}") from e
    if df.empty or not len(df.columns):
        raise HTTPException(400, "File CSV không có dữ liệu.")
    return df


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)


class PredictRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=MAX_BATCH_ROWS)


@app.get("/api/health")
def health():
    """`ready` = model đã nạp; `metrics_ready` = đã đánh giá xong tập test."""
    if _state["error"] is None:
        try:
            inference.check_artifacts()
        except inference.MissingArtifactError as e:
            _state["error"] = str(e)
    return {
        "ready": inference.get_model.cache_info().currsize > 0,
        "metrics_ready": inference.evaluate_test_set.cache_info().currsize > 0,
        "error": _state["error"],
    }


@app.get("/api/model")
def model_info():
    """Cấu hình model và kết quả trên tập test (chờ nếu đang tính lần đầu)."""
    _require_ready()
    model, _, config = inference.get_model()
    return {
        "config": config,
        "n_params": sum(p.numel() for p in model.parameters()),
        "labels": LABELS,
        "test_metrics": _metrics_json(_test_metrics()),
    }


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    """Phân tích 1 bình luận: nhãn, xác suất, token sau tách từ và attention (layer × head × T × T)."""
    _require_ready()
    result = inference.analyze(req.text)
    if result is None:
        return {"empty": True}
    probs = result["probs"]
    return {
        "empty": False,
        "prediction": LABELS[int(np.argmax(probs))],
        "probs": probs.round(4).tolist(),
        "cleaned": result["cleaned"],
        "segmented": result["segmented"],
        "n_tokens": result["n_tokens"],
        "max_len": result["max_len"],
        "tokens": result["tokens"],
        "is_unk": result["is_unk"],
        "attention": result["attn"].round(4).tolist(),
    }


@app.post("/api/predict")
def predict(req: PredictRequest):
    """Phân loại nhiều bình luận (không kèm attention)."""
    _require_ready()
    rows, summary = _classify(req.texts, None)
    for row, text in zip(rows, req.texts):
        row["text"] = text
    return {"rows": rows, "summary": summary}


@app.get("/api/samples")
def samples(n: int = Query(10, ge=1, le=50), only_toxic: bool = False):
    """Lấy ngẫu nhiên `n` câu từ tập test, kèm nhãn thật và dự đoán."""
    _require_ready()
    df = inference.load_test_set()
    pool = df[df["label_id"] != 0] if only_toxic else df
    picked = pool.sample(n).reset_index(drop=True)
    texts = picked["free_text"].astype(str).tolist()
    rows, summary = _classify(texts, picked["label_id"])
    for row, text in zip(rows, texts):
        row["text"] = text
    return {"rows": rows, "summary": summary}


@app.post("/api/csv/inspect")
async def csv_inspect(file: UploadFile = File(...)):
    """Đọc header của file CSV để người dùng chọn cột văn bản và cột nhãn."""
    df = await _read_upload(file)
    cols = [str(c) for c in df.columns]
    text_col = "free_text" if "free_text" in cols else cols[0]
    label_col = next((c for c in ("label_id", "label", "labels") if c in cols), None)
    return {"columns": cols, "n_rows": len(df), "text_col": text_col, "label_col": label_col,
            "preview": _records(df.head(5))}


@app.post("/api/csv/predict")
async def csv_predict(file: UploadFile = File(...), text_col: str = Form(...),
                      label_col: str | None = Form(None)):
    """Phân loại toàn bộ file CSV; nếu có cột nhãn thì tính thêm accuracy, macro-F1, confusion matrix."""
    _require_ready()
    df = await _read_upload(file)
    df.columns = [str(c) for c in df.columns]
    for col in (text_col, label_col):
        if col and col not in df.columns:
            raise HTTPException(400, f"Không có cột `{col}` trong file.")
    texts = df[text_col].fillna("").astype(str).tolist()
    # Hàm async chạy trên event loop, nên đẩy phần tính toán nặng sang threadpool.
    rows, summary = await run_in_threadpool(_classify, texts, df[label_col] if label_col else None)
    records = _records(df)
    for row, rec in zip(rows, records):
        row["data"] = rec
    return {"columns": list(df.columns), "text_col": text_col, "label_col": label_col,
            "rows": rows, "summary": summary}


if WEB_DIST.exists():
    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")
