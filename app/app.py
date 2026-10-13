"""Demo phát hiện bình luận độc hại tiếng Việt bằng Transformer Encoder tự cài đặt.

Chạy từ thư mục gốc project:
    make start      (hoặc: streamlit run app/app.py)
"""

import html
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.data import LABEL_NAMES  # noqa: E402
from core.preprocessing import clean_text, segment_text  # noqa: E402
from model import load_model  # noqa: E402

MODEL_DIR = ROOT / "models" / "transformer-scratch-vihsd"
TEST_CSV = ROOT / "data" / "vihsd" / "vihsd" / "test.csv"
LOGO = Path(__file__).resolve().parent / "assets" / "hcmute_logo.png"

TEAM_NAME = "Nhóm 5"
SUPERVISOR = "PGS.TS. Hoàng Văn Dũng"
MEMBERS = [
    ("2611318", "Vòng Vĩnh Shìn"),
    ("2611302", "Trần Nhật Duật"),
    ("2611320", "Hồ Đức Thắng"),
]

LABELS = [LABEL_NAMES[i] for i in sorted(LABEL_NAMES)]
LABEL_COLORS = {"CLEAN": "#16a34a", "OFFENSIVE": "#f59e0b", "HATE": "#dc2626"}
LABEL_VI = {"CLEAN": "Bình thường", "OFFENSIVE": "Xúc phạm", "HATE": "Thù ghét"}
ACCENT_RGB = "21,101,192"  # xanh HCM-UTE, trùng theme.primaryColor trong .streamlit/config.toml

EXAMPLES = [
    "Video này hay quá, cảm ơn bạn đã chia sẻ nhé",
    "Thằng ngu này nói chuyện chán vãi",
    "Bọn mày là lũ súc vật không hơn không kém",
]

st.set_page_config(page_title="ViHSD Transformer Demo", page_icon=str(LOGO), layout="wide")

st.html(f"""
<style>
[data-testid="stMainBlockContainer"] {{ padding-top: 1.5rem; padding-bottom: 2rem; }}
[data-testid="stSidebarHeader"] {{ height: 1rem; padding: 0; }}
[data-testid="stSidebarUserContent"] {{ padding-top: 0.5rem; }}
.app-title {{ font-size: 2rem !important; font-weight: 800; line-height: 1.25; }}
.app-subtitle {{ opacity: 0.65; margin: 0.15rem 0 0.4rem; }}
.school {{ text-align: center; line-height: 1.35; font-size: 0.78rem; margin-bottom: 0.25rem; }}
.team-card {{ background: rgba({ACCENT_RGB},0.08); border-left: 4px solid rgb({ACCENT_RGB});
              border-radius: 8px; padding: 0.75rem 0.9rem; font-size: 0.92rem; }}
.team-card .team {{ font-weight: 800; font-size: 1.05rem; margin-bottom: 0.4rem; }}
.team-card .member {{ display: flex; justify-content: space-between; padding: 2px 0; }}
.team-card .mshv {{ font-family: monospace; opacity: 0.7; }}
.team-card .gvhd {{ margin-top: 0.5rem; padding-top: 0.5rem; border-top: 1px solid rgba({ACCENT_RGB},0.25); }}
.pred-badge {{ display: inline-block; color: white; padding: 6px 18px; border-radius: 999px;
               font-weight: 800; font-size: 1.25rem; }}
.chip {{ display: inline-block; margin: 3px; padding: 3px 9px; border-radius: 6px; }}
.step {{ display: inline-block; margin: 3px 0; padding: 4px 10px; border-radius: 6px;
         background: rgba({ACCENT_RGB},0.1); border: 1px solid rgba({ACCENT_RGB},0.3); }}
</style>
""")


# ---------------------------------------------------------------------------
# Model & dữ liệu
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Đang nạp model...")
def get_model():
    model, word2idx, config = load_model(MODEL_DIR)
    segment_text("khởi động")  # nạp sẵn model tách từ của underthesea
    return model, word2idx, config


@st.cache_data
def load_test_set():
    return pd.read_csv(TEST_CSV)


def read_uploaded_csv(uploaded):
    """Đọc CSV upload; file lưu bằng Excel trên Windows thường dùng bảng mã cp1258/cp1252 thay vì UTF-8."""
    for encoding in ("utf-8-sig", "cp1258", "cp1252"):
        try:
            uploaded.seek(0)
            df = pd.read_csv(uploaded, encoding=encoding)
        except UnicodeDecodeError:
            continue
        # cp1258 lưu dấu tiếng Việt dạng tổ hợp (NFD) → chuẩn hoá NFC cho khớp vocab.
        for col in df.select_dtypes("object"):
            df[col] = df[col].map(lambda v: unicodedata.normalize("NFC", v) if isinstance(v, str) else v)
        return df
    st.error("Không đọc được file CSV — hãy lưu lại với bảng mã UTF-8.")
    st.stop()


def tokenize(text: str, word2idx: dict, max_len: int):
    """Làm sạch + tách từ + mã hoá giống hệt pipeline lúc train."""
    segmented = segment_text(clean_text(text))
    tokens = segmented.split()[:max_len]
    ids = [word2idx.get(tok, word2idx["<unk>"]) for tok in tokens]
    return tokens, ids


@torch.no_grad()
def predict_batch(texts: list[str], batch_size: int = 64):
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


@st.cache_data(show_spinner="Đang đánh giá model trên tập test (chỉ chạy lần đầu)...")
def evaluate_test_set():
    """Chạy model trên toàn bộ tập test để hiển thị số liệu thật của checkpoint hiện tại."""
    df = load_test_set()
    probs = predict_batch(df["free_text"].astype(str).tolist())
    valid = ~np.isnan(probs).any(axis=1)
    y_true = df.loc[valid, "label_id"].map(LABEL_NAMES)
    y_pred = [LABELS[i] for i in probs[valid].argmax(axis=1)]
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    return {
        "n": int(valid.sum()),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=LABELS),
        "per_class": pd.DataFrame({"Precision": p, "Recall": r, "F1": f, "Support": s}, index=LABELS),
        "cm": confusion_matrix(y_true, y_pred, labels=LABELS),
    }


@torch.no_grad()
def analyze(text: str):
    """Dự đoán 1 câu, kèm token và attention của mọi layer (chỉ phần token thật)."""
    model, word2idx, config = get_model()
    tokens, ids = tokenize(text, word2idx, config["max_len"])
    if not ids:
        return None

    logits, attentions = model(torch.tensor([ids], dtype=torch.long))
    probs = torch.softmax(logits, dim=-1)[0].numpy()
    attn = np.stack([a[0].numpy() for a in attentions])  # (L, H, T, T)
    is_unk = [i == word2idx["<unk>"] for i in ids]
    return {"tokens": tokens, "is_unk": is_unk, "probs": probs, "attn": attn}


# ---------------------------------------------------------------------------
# Thành phần hiển thị
# ---------------------------------------------------------------------------

def probs_chart(probs: np.ndarray):
    fig = go.Figure(go.Bar(
        x=probs, y=[f"{l} · {LABEL_VI[l]}" for l in LABELS], orientation="h",
        marker_color=[LABEL_COLORS[l] for l in LABELS],
        text=[f"{p:.1%}" for p in probs], textposition="outside", cliponaxis=False,
    ))
    fig.update_layout(
        height=170, margin=dict(l=0, r=40, t=0, b=0),
        xaxis=dict(range=[0, 1], visible=False), yaxis=dict(autorange="reversed"),
    )
    return fig


def confusion_chart(cm: np.ndarray, height: int = 300):
    fig = px.imshow(cm, x=LABELS, y=LABELS, text_auto=True, color_continuous_scale="Blues",
                    labels=dict(x="Dự đoán", y="Thực tế", color="Số lượng"))
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=10, b=0), coloraxis_showscale=False)
    return fig


def token_chips(tokens, is_unk, scores) -> str:
    """Token sau tách từ, tô nền theo mức attention nhận được."""
    scores = scores / scores.max()
    chips = []
    for tok, unk, score in zip(tokens, is_unk, scores):
        border = "2px dashed #9ca3af" if unk else f"1px solid rgba({ACCENT_RGB},0.2)"
        title = "Không có trong vocab → &lt;unk&gt;" if unk else f"Attention nhận được: {score:.2f}"
        chips.append(
            f"<span class='chip' title='{title}' style='border:{border};"
            f"background:rgba({ACCENT_RGB},{0.06 + 0.7 * float(score):.2f})'>{html.escape(tok.replace('_', ' '))}</span>"
        )
    return "<div style='line-height:2.3'>" + "".join(chips) + "</div>"


def render_prediction(result, true_label: str | None = None, key: str = "main"):
    probs = result["probs"]
    pred = LABELS[int(np.argmax(probs))]
    st.markdown(
        f"<span class='pred-badge' style='background:{LABEL_COLORS[pred]}'>{pred} · {LABEL_VI[pred]}</span>",
        unsafe_allow_html=True,
    )
    info = f"Độ tin cậy **{probs.max():.1%}**"
    if true_label is not None:
        info += f" · Nhãn thật **{true_label}** " + ("✅ Đúng" if pred == true_label else "❌ Sai")
    st.markdown(info)
    st.plotly_chart(probs_chart(probs), width="stretch", key=f"probs_{key}",
                    config={"displayModeBar": False})


def render_attention(result, key: str = "main"):
    tokens, attn = result["tokens"], result["attn"]
    n_layers, n_heads = attn.shape[:2]

    head_col, layer_col, sel_col = st.columns([2, 1, 1], vertical_alignment="bottom")
    head_col.markdown("#### 🔎 Self-attention")
    layer = layer_col.selectbox(
        "Layer", range(n_layers), index=n_layers - 1,
        format_func=lambda i: f"Layer {i + 1}", key=f"layer_{key}",
    )
    head = sel_col.selectbox(
        "Head", ["avg", *range(n_heads)],
        format_func=lambda h: "Trung bình các head" if h == "avg" else f"Head {h + 1}",
        key=f"head_{key}",
    )
    matrix = attn[layer].mean(axis=0) if head == "avg" else attn[layer, head]

    # Mức attention trung bình mỗi token nhận được từ các token khác (trung bình theo cột).
    st.markdown(token_chips(tokens, result["is_unk"], matrix.mean(axis=0)), unsafe_allow_html=True)
    st.caption("Nền càng đậm → token càng được các token khác chú ý nhiều. "
               "Viền đứt = từ không có trong vocab (mã hoá thành `<unk>`).")

    labels = [f"{i}:{t.replace('_', ' ')}" for i, t in enumerate(tokens)]
    fig = px.imshow(
        matrix, x=labels, y=labels, color_continuous_scale="Blues", zmin=0,
        labels=dict(x="Key (được chú ý)", y="Query (đang chú ý)", color="Weight"),
        aspect="auto",
    )
    fig.update_traces(hovertemplate="Query: %{y}<br>Key: %{x}<br>Weight: %{z:.3f}<extra></extra>")
    size = max(360, min(680, 28 * len(tokens) + 140))
    fig.update_layout(height=size, margin=dict(l=0, r=0, t=10, b=0), xaxis_tickangle=-45)
    st.plotly_chart(fig, width="stretch", key=f"heatmap_{key}")
    st.caption("Mỗi hàng là phân phối softmax của 1 query trên toàn bộ key (tổng hàng = 1). "
               "Token `<pad>` đã bị mask nên không xuất hiện.")


def render_full_analysis(text: str, true_label: str | None = None, key: str = "main"):
    result = analyze(text)
    if result is None:
        st.warning("Câu trống sau khi làm sạch (chỉ có URL / mention / ký tự đặc biệt).")
        return
    with st.container(border=True):
        render_prediction(result, true_label, key)
    with st.container(border=True):
        render_attention(result, key)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

if not (MODEL_DIR / "model_state_dict.pt").exists():
    st.error(f"Không tìm thấy checkpoint tại `{MODEL_DIR}`. Hãy chạy notebook "
             "`src/vihsd_transformer_from_scratch.ipynb` để train và lưu model trước.")
    st.stop()
if not TEST_CSV.exists():
    st.error(f"Không tìm thấy tập test tại `{TEST_CSV}`. Hãy chạy notebook "
             "`src/vihsd_transformer_from_scratch.ipynb` để tải dataset ViHSD trước.")
    st.stop()

model, word2idx, config = get_model()
n_params = sum(p.numel() for p in model.parameters())

THEMES = {"☀️ Sáng": "light", "🌙 Tối": "dark"}


def apply_theme():
    # Streamlit chưa có API public để đổi theme lúc chạy; menu ⋮ mặc định đã bị ẩn
    # (client.toolbarMode = "minimal") nên đổi qua config, áp dụng ở lần chạy lại kế tiếp.
    st._config.set_option("theme.base", THEMES[st.session_state["theme"]])


with st.sidebar:
    _, logo_col, _ = st.columns([1, 2, 1])
    logo_col.image(str(LOGO), width="stretch")
    st.markdown(
        "<div class='school'><b>TRƯỜNG ĐẠI HỌC CÔNG NGHỆ KỸ THUẬT TP.HCM</b><br>"
        "Khoa Công nghệ Thông tin</div>",
        unsafe_allow_html=True,
    )
    members_html = "".join(
        f"<div class='member'><span>{name}</span><span class='mshv'>{mshv}</span></div>"
        for mshv, name in MEMBERS
    )
    st.markdown(
        f"<div class='team-card'><div class='team'>👥 {TEAM_NAME}</div>{members_html}"
        f"<div class='gvhd'><b>GVHD:</b> {SUPERVISOR}</div></div>",
        unsafe_allow_html=True,
    )

    st.divider()
    with st.popover("⚙️ Cài đặt", width="stretch"):
        st.radio("Giao diện", list(THEMES), key="theme", horizontal=True, on_change=apply_theme)
        c1, c2 = st.columns(2)
        if c1.button("🔄 Chạy lại", width="stretch"):
            st.rerun()
        if c2.button("🧹 Xoá cache", width="stretch",
                     help="Xoá cache dữ liệu và nạp lại model ở lần chạy kế tiếp"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

metrics = evaluate_test_set()

st.markdown(
    "<div class='app-title'>🛡️ Phát hiện bình luận độc hại tiếng Việt</div>"
    "<div class='app-subtitle'>Ứng dụng kỹ thuật Transformer · Transformer Encoder tự cài đặt, "
    "huấn luyện từ đầu trên dataset ViHSD</div>",
    unsafe_allow_html=True,
)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Accuracy (test)", f"{metrics['accuracy']:.2%}", border=True)
m2.metric("Macro-F1 (test)", f"{metrics['macro_f1']:.4f}", border=True)
m3.metric("Layer × Head", f"{config['n_layers']} × {config['n_heads']}", border=True)
m4.metric("Số tham số", f"{n_params / 1e6:.2f}M", border=True)

tab_single, tab_samples, tab_batch, tab_model = st.tabs(
    ["💬 Phân tích bình luận", "🎲 Ví dụ từ tập test", "📄 Phân loại hàng loạt", "🧠 Mô hình"]
)


# ---------------------------------------------------------------------------
# Tab 1: phân tích 1 bình luận
# ---------------------------------------------------------------------------

with tab_single:
    col_in, col_out = st.columns([5, 7], gap="medium")
    with col_in, st.container(border=True, height="stretch"):
        choice = st.selectbox("Câu mẫu", [*EXAMPLES, ""], format_func=lambda s: s or "✍️ Tự nhập",
                              on_change=lambda: st.session_state.pop("single_query", None))
        text = st.text_area("Bình luận", value=choice, height=110,
                            placeholder="Nhập một bình luận tiếng Việt...")
        if st.button("🔍 Kiểm tra", type="primary", width="stretch"):
            if text.strip():
                st.session_state["single_query"] = text
            else:
                st.session_state.pop("single_query", None)
                st.warning("Hãy nhập một bình luận trước khi kiểm tra.")

    # Lưu câu đã kiểm tra để kết quả không mất khi đổi layer/head của heatmap.
    query = st.session_state.get("single_query")
    result = analyze(query) if query else None
    with col_out, st.container(border=True, height="stretch"):
        st.markdown("**Kết quả dự đoán**")
        if result is not None:
            render_prediction(result, key="single")
        elif query:
            st.warning("Câu trống sau khi làm sạch (chỉ có URL / mention / ký tự đặc biệt).")
        else:
            st.info("Chọn câu mẫu hoặc nhập bình luận, rồi bấm **Kiểm tra**.")

    if result is not None:
        with st.container(border=True):
            render_attention(result, key="single")


# ---------------------------------------------------------------------------
# Tab 2: ví dụ từ tập test
# ---------------------------------------------------------------------------

with tab_samples:
    test_df = load_test_set()
    c1, c2, c3 = st.columns([1, 2, 1], vertical_alignment="bottom")
    n = c1.number_input("Số câu", 5, 50, 10)
    only_toxic = c2.checkbox("Chỉ lấy câu độc hại (OFFENSIVE/HATE)",
                             help="CLEAN chiếm ~82% tập test nên lấy ngẫu nhiên thường ra toàn câu sạch")
    if c3.button("🎲 Lấy mẫu ngẫu nhiên", type="primary", width="stretch") or "samples" not in st.session_state:
        pool = test_df[test_df["label_id"] != 0] if only_toxic else test_df
        st.session_state["samples"] = pool.sample(int(n)).reset_index(drop=True)

    samples = st.session_state["samples"].copy()
    probs = predict_batch(samples["free_text"].tolist())
    samples["Nhãn thật"] = samples["label_id"].map(LABEL_NAMES)
    samples["Dự đoán"] = [LABELS[int(np.argmax(p))] if not np.isnan(p).any() else "—" for p in probs]
    samples["Tin cậy"] = np.nanmax(probs, axis=1)
    samples["Kết quả"] = np.where(samples["Nhãn thật"] == samples["Dự đoán"], "✅", "❌")

    n_correct = (samples["Kết quả"] == "✅").sum()
    st.markdown(f"Đúng **{n_correct}/{len(samples)}** câu · bấm vào 1 dòng để xem phân tích chi tiết")
    event = st.dataframe(
        samples[["free_text", "Nhãn thật", "Dự đoán", "Tin cậy", "Kết quả"]].rename(columns={"free_text": "Bình luận"}),
        width="stretch", hide_index=True, on_select="rerun", selection_mode="single-row",
        column_config={"Tin cậy": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)},
    )
    rows = event.selection.rows
    idx = rows[0] if rows else 0
    st.markdown(f"**Câu đang phân tích:** {samples.at[idx, 'free_text']}")
    render_full_analysis(samples.at[idx, "free_text"], true_label=samples.at[idx, "Nhãn thật"], key="sample")


# ---------------------------------------------------------------------------
# Tab 3: phân loại hàng loạt
# ---------------------------------------------------------------------------

with tab_batch:
    df = None
    text_col, label_col = "Bình luận", None

    with st.container(border=True):
        mode = st.segmented_control("Nguồn dữ liệu", ["Dán nhiều dòng", "Upload CSV"], default="Dán nhiều dòng")
        if mode == "Upload CSV":
            uploaded = st.file_uploader(
                "Chọn file CSV", type="csv",
                help="Cần 1 cột văn bản. Nếu có cột nhãn (0/1/2 hoặc CLEAN/OFFENSIVE/HATE), app tính thêm "
                     "accuracy, macro-F1 và confusion matrix. Có thể thử với data/vihsd/vihsd/test.csv.",
            )
            if uploaded is not None:
                df = read_uploaded_csv(uploaded)
                cols = list(df.columns)
                c1, c2 = st.columns(2)
                text_col = c1.selectbox("Cột văn bản", cols,
                                        index=cols.index("free_text") if "free_text" in cols else 0)
                label_opts = ["(không có)", *cols]
                label_col = c2.selectbox("Cột nhãn thật", label_opts,
                                         index=label_opts.index("label_id") if "label_id" in cols else 0)
                label_col = None if label_col == "(không có)" else label_col
        else:
            raw = st.text_area("Mỗi dòng là một bình luận", height=160,
                               placeholder="Bình luận 1\nBình luận 2\n...")
            lines = [l for l in raw.splitlines() if l.strip()]
            if lines:
                df = pd.DataFrame({text_col: lines})

        clicked = st.button(f"🚀 Phân loại {len(df):,} bình luận" if df is not None else "🚀 Phân loại",
                            type="primary", disabled=df is None)

    if clicked:
        with st.spinner("Đang tách từ và dự đoán..."):
            probs = predict_batch(df[text_col].astype(str).tolist())
        valid = ~np.isnan(probs).any(axis=1)
        result = df.copy()
        result["prediction"] = [LABELS[int(np.argmax(p))] if ok else "EMPTY" for p, ok in zip(probs, valid)]
        for i, l in enumerate(LABELS):
            result[f"p_{l}"] = probs[:, i].round(4)
        st.session_state["batch_result"] = (result, text_col, label_col)

    if "batch_result" in st.session_state:
        result, text_col, label_col = st.session_state["batch_result"]
        counts = result["prediction"].value_counts().reindex(LABELS, fill_value=0)

        c1, c2 = st.columns(2)
        with c1, st.container(border=True, height="stretch"):
            st.markdown("**Phân bố dự đoán**")
            fig = px.pie(values=counts.values, names=counts.index, color=counts.index,
                         color_discrete_map=LABEL_COLORS, hole=0.5)
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, width="stretch")

        if label_col is not None:
            truth = result[label_col]
            if pd.api.types.is_numeric_dtype(truth):
                truth = truth.map(LABEL_NAMES)
            mask = result["prediction"].isin(LABELS) & truth.isin(LABELS)
            y_true, y_pred = truth[mask], result.loc[mask, "prediction"]
            with c2, st.container(border=True, height="stretch"):
                st.markdown("**Đánh giá so với nhãn thật**")
                k1, k2 = st.columns(2)
                k1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.2%}")
                k2.metric("Macro-F1", f"{f1_score(y_true, y_pred, average='macro', labels=LABELS):.4f}")
                st.plotly_chart(confusion_chart(confusion_matrix(y_true, y_pred, labels=LABELS), 230),
                                width="stretch")
            result["correct"] = np.where(truth == result["prediction"], "✅", "❌")
        else:
            with c2, st.container(border=True, height="stretch"):
                st.markdown(f"**Thống kê** ({len(result):,} bình luận)")
                for l, col in zip(LABELS, st.columns(len(LABELS))):
                    col.metric(f"{l} · {LABEL_VI[l]}", f"{counts[l]:,}")
                    col.caption(f"{counts[l] / len(result):.1%}")

        st.dataframe(result, width="stretch", hide_index=True)
        st.download_button("⬇️ Tải kết quả CSV", result.to_csv(index=False).encode("utf-8-sig"),
                           file_name="ket_qua_phan_loai.csv", mime="text/csv")


# ---------------------------------------------------------------------------
# Tab 4: thông tin mô hình
# ---------------------------------------------------------------------------

with tab_model:
    c1, c2 = st.columns(2, gap="medium")
    with c1, st.container(border=True, height="stretch"):
        st.markdown("**Kiến trúc**")
        st.dataframe(pd.DataFrame({
            "Thành phần": ["Số encoder layer", "Số attention head", "d_model", "d_ff (feed-forward)",
                           "Độ dài tối đa", "Kích thước vocab", "Tổng tham số"],
            "Giá trị": [str(config["n_layers"]), str(config["n_heads"]), str(config["d_model"]),
                        str(config["d_ff"]), f"{config['max_len']} token", f"{config['vocab_size']:,}",
                        f"{n_params:,}"],
        }), hide_index=True, width="stretch")

        st.markdown("**Pipeline xử lý**")
        steps = ["Làm sạch", "Tách từ (underthesea)", "Mã hoá token", "Embedding + Positional Encoding",
                 f"{config['n_layers']} × Encoder (Multi-head Attention + FFN)", "Mean pooling",
                 "Linear + Softmax"]
        st.markdown(" → ".join(f"<span class='step'>{s}</span>" for s in steps), unsafe_allow_html=True)

        st.markdown("**Nhãn**")
        for l in LABELS:
            st.markdown(f"<span style='color:{LABEL_COLORS[l]}'>●</span> **{l}** — {LABEL_VI[l]}",
                        unsafe_allow_html=True)

    with c2, st.container(border=True, height="stretch"):
        st.markdown(f"**Kết quả trên tập test** ({metrics['n']:,} bình luận)")
        st.dataframe(
            metrics["per_class"].style.format({"Precision": "{:.4f}", "Recall": "{:.4f}", "F1": "{:.4f}"}),
            width="stretch",
        )
        st.plotly_chart(confusion_chart(metrics["cm"]), width="stretch")
