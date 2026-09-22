# Ứng dụng kỹ thuật Transformer trong bài toán phát hiện bình luận độc hại

Đồ án minh hoạ và triển khai kiến trúc **Transformer** cho bài toán phát hiện bình luận độc hại tiếng Việt,
trên dataset **ViHSD** (Vietnamese Hate Speech Detection).

## Cấu trúc project

```
demo-transformer/
├── requirements.txt
├── src/
│   ├── demo.ipynb                              # Minh hoạ cơ chế self-attention (NumPy, từng bước)
│   ├── vihsd_phobert_toxic_detection.ipynb      # Fine-tune PhoBERT (pretrained) trên ViHSD
│   └── vihsd_transformer_from_scratch.ipynb     # Transformer Encoder tự viết bằng PyTorch (không pretrained)
├── data/    # dữ liệu ViHSD tải về (tự sinh khi chạy notebook, không commit)
└── models/  # checkpoint model đã train (tự sinh khi chạy notebook, không commit)
```

## Các notebook

| Notebook | Mô tả |
|---|---|
| `demo.ipynb` | Cài đặt self-attention từ đầu bằng NumPy trên 1 câu ví dụ, minh hoạ Q/K/V, attention score, softmax, output — phục vụ giải thích lý thuyết. |
| `vihsd_phobert_toxic_detection.ipynb` | Fine-tune `vinai/phobert-base` (pretrained) cho phân loại 3 lớp CLEAN/OFFENSIVE/HATE. Có EDA, tiền xử lý (làm sạch + tách từ), xử lý mất cân bằng lớp, đánh giá chi tiết (classification report, confusion matrix, error analysis). |
| `vihsd_transformer_from_scratch.ipynb` | Tự cài đặt kiến trúc Transformer Encoder hoàn toàn bằng PyTorch thuần (embedding, positional encoding, multi-head self-attention, feed-forward, residual + layer norm), huấn luyện từ trọng số ngẫu nhiên (không pretrained), dùng để đối chiếu với kết quả fine-tune PhoBERT. |

## Dataset

**ViHSD** — 33.400 bình luận mạng xã hội tiếng Việt, gán nhãn: `CLEAN` (0), `OFFENSIVE` (1), `HATE` (2).
Dataset được các notebook tự động tải từ repo gốc [`sonlam1102/vihsd`](https://github.com/sonlam1102/vihsd)
vào thư mục `data/vihsd/` — không cần tải thủ công. Dataset chỉ dùng cho mục đích nghiên cứu; vui lòng trích
dẫn paper gốc khi sử dụng (xem phần cuối mỗi notebook).

## Cài đặt môi trường

Yêu cầu Python ≥ 3.10.

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Đăng ký venv làm Jupyter kernel (để chọn trong VS Code / Jupyter):

```bash
python -m ipykernel install --user --name demo-transformer --display-name "Python (demo-transformer)"
```

Sau đó mở notebook trong `src/` và chọn kernel **"Python (demo-transformer)"**.

## Chạy notebook

1. `src/demo.ipynb` — chạy độc lập, không cần dataset, chạy nhanh.
2. `src/vihsd_phobert_toxic_detection.ipynb` — chạy tuần tự từ trên xuống. Tự tải dataset ở phần 2.
   Có Apple Silicon (MPS) sẽ tự tăng tốc; nếu chỉ có CPU, quá trình fine-tune sẽ chậm hơn.
3. `src/vihsd_transformer_from_scratch.ipynb` — chạy tuần tự từ trên xuống, dùng lại dataset đã tải ở bước 2
   (hoặc tự tải nếu chưa có). Có cờ để chỉnh số epoch/kiến trúc ở phần cấu hình model.

## Ghi chú

- `data/` và `models/` không được commit (xem `.gitignore`) — sẽ tự sinh khi chạy notebook.
- Nếu link tải dataset gốc bị thay đổi, xem hướng dẫn thay thế trong README của repo
  [`sonlam1102/vihsd`](https://github.com/sonlam1102/vihsd).
