# Ứng dụng kỹ thuật Transformer trong bài toán phát hiện bình luận độc hại

Đồ án minh hoạ và triển khai kiến trúc **Transformer** cho bài toán phát hiện bình luận độc hại tiếng Việt,
trên dataset **ViHSD** (Vietnamese Hate Speech Detection).

## Cấu trúc project

```
demo-transformer/
├── requirements.txt
├── Makefile
├── apps/
│   ├── api/    # Backend FastAPI: main.py (endpoint), inference.py (suy luận), model.py (kiến trúc)
│   └── web/    # Frontend React + TypeScript + Tailwind (Vite)
├── src/
│   ├── vihsd_phobert_toxic_detection.ipynb      # Fine-tune PhoBERT (pretrained) trên ViHSD
│   ├── vihsd_transformer_from_scratch.ipynb     # Transformer Encoder tự viết bằng PyTorch (không pretrained)
│   └── core/                                    # Hàm dùng chung: tải dữ liệu, tiền xử lý, đánh giá
├── data/    # dữ liệu ViHSD tải về (tự sinh khi chạy notebook, không commit)
└── models/  # checkpoint model đã train (tự sinh khi chạy notebook, không commit)
```

## Các notebook

| Notebook | Mô tả |
|---|---|
| `vihsd_phobert_toxic_detection.ipynb` | Fine-tune `vinai/phobert-base` (pretrained) cho phân loại 3 lớp CLEAN/OFFENSIVE/HATE. Có EDA, tiền xử lý (làm sạch + tách từ), xử lý mất cân bằng lớp, đánh giá chi tiết (classification report, confusion matrix, error analysis). |
| `vihsd_transformer_from_scratch.ipynb` | Tự cài đặt kiến trúc Transformer Encoder hoàn toàn bằng PyTorch thuần (embedding, positional encoding, multi-head self-attention, feed-forward, residual + layer norm), huấn luyện từ trọng số ngẫu nhiên (không pretrained), dùng để đối chiếu với kết quả fine-tune PhoBERT. |

## Dataset

**ViHSD** — 33.400 bình luận mạng xã hội tiếng Việt, gán nhãn: `CLEAN` (0), `OFFENSIVE` (1), `HATE` (2).
Dataset được các notebook tự động tải từ repo gốc [`sonlam1102/vihsd`](https://github.com/sonlam1102/vihsd)
vào thư mục `data/vihsd/` — không cần tải thủ công. Dataset chỉ dùng cho mục đích nghiên cứu; vui lòng trích
dẫn paper gốc khi sử dụng (xem phần cuối mỗi notebook).

## Cài đặt môi trường

Yêu cầu Python ≥ 3.10. Có thể dùng `make install` thay cho bốn lệnh bên dưới.

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

1. `src/vihsd_phobert_toxic_detection.ipynb` — chạy tuần tự từ trên xuống. Tự tải dataset ở phần 2.
   Có Apple Silicon (MPS) sẽ tự tăng tốc; nếu chỉ có CPU, quá trình fine-tune sẽ chậm hơn.
2. `src/vihsd_transformer_from_scratch.ipynb` — chạy tuần tự từ trên xuống, dùng lại dataset đã tải ở bước 1
   (hoặc tự tải nếu chưa có). Có cờ để chỉnh số epoch/kiến trúc ở phần cấu hình model.

## Chạy ứng dụng demo (FastAPI + React)

App gồm backend FastAPI (`apps/api`) và giao diện React (`apps/web`). App không tự huấn luyện model. Nó
đọc checkpoint và tập test do notebook from-scratch sinh ra, nên lần đầu cần làm đủ các bước dưới đây theo
đúng thứ tự. Mọi lệnh đều chạy từ thư mục gốc project.

Yêu cầu: Python ≥ 3.10 và Node.js ≥ 20 (kèm npm).

**Bước 1. Cài môi trường** (chỉ làm một lần)

```bash
make install   # tạo .venv, cài requirements.txt và thư viện của apps/web
```

Windows không có sẵn `make`, chạy trực tiếp (PowerShell hoặc cmd):

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
npm --prefix apps/web ci
```

**Bước 2. Đăng ký kernel cho notebook** (chỉ làm một lần)

```bash
.venv/bin/python -m ipykernel install --user --name demo-transformer --display-name "Python (demo-transformer)"
# Windows: .venv\Scripts\python -m ipykernel install --user --name demo-transformer --display-name "Python (demo-transformer)"
```

**Bước 3. Chạy notebook để tải dataset và huấn luyện model**

Mở `src/vihsd_transformer_from_scratch.ipynb` trong VS Code hoặc Jupyter, chọn kernel
**"Python (demo-transformer)"** rồi chạy toàn bộ notebook (Run All). Quá trình mất khoảng 4 phút trên
Apple M1 Pro (MPS) và lâu hơn nếu chỉ có CPU.

Notebook dùng đường dẫn tương đối `../data` và `../models`, nên phải chạy với thư mục làm việc là `src/`
(mặc định khi mở notebook trong VS Code hoặc Jupyter). Sau khi chạy xong phải có đủ các file sau:

```
data/vihsd/vihsd/test.csv
models/transformer-scratch-vihsd/model_state_dict.pt
models/transformer-scratch-vihsd/vocab.json
models/transformer-scratch-vihsd/config.json
```

**Bước 4. Chạy app**

```bash
make start                # build giao diện rồi chạy app, mặc định cổng 8000
make start PORT=8600      # đổi cổng
```

Windows:

```powershell
npm --prefix apps/web run build
.venv\Scripts\python -m uvicorn main:app --app-dir apps/api --port 8000
```

Lần đầu khởi động, server mất vài giây để nạp model và đánh giá trên tập test. Nếu thiếu checkpoint hoặc
tập test, giao diện sẽ báo lỗi và chỉ ra file còn thiếu. Khi đó quay lại Bước 3. Những lần chạy sau chỉ cần
Bước 4, trừ khi muốn huấn luyện lại model.

App mở tại http://localhost:8000, gồm 3 tab:

| Tab | Chức năng |
|---|---|
| Mô phỏng bình luận | Khung bình luận dạng chat: nhập liên tục, mỗi bình luận được gắn nhãn ngay. Có thể làm mờ bình luận độc hại. Bấm vào bình luận để mở pop-up chi tiết (tiền xử lý, xác suất, attention). Lịch sử lưu trong trình duyệt |
| Ví dụ từ tập test | Lấy mẫu ngẫu nhiên từ `test.csv`, so sánh dự đoán với nhãn thật, phân tích chi tiết từng câu |
| Mô hình | Kiến trúc, pipeline xử lý, kết quả từng lớp và confusion matrix trên tập test |

### API

Tài liệu API (Swagger) tự sinh tại http://localhost:8000/docs.

| Endpoint | Chức năng |
|---|---|
| `GET /api/health` | Trạng thái nạp model và lỗi thiếu file (nếu có) |
| `GET /api/model` | Cấu hình, số tham số, kết quả trên tập test |
| `POST /api/analyze` | Phân tích một câu, kèm token và attention |
| `POST /api/predict` | Phân loại nhiều câu, tối đa 10.000 câu mỗi lần |
| `GET /api/samples?n=10&only_toxic=false` | Lấy câu ngẫu nhiên từ tập test, kèm dự đoán |
| `POST /api/csv/inspect` | Đọc danh sách cột của file CSV upload |
| `POST /api/csv/predict` | Phân loại file CSV, tính chỉ số nếu có cột nhãn |

### Phát triển giao diện

```bash
make dev       # API ở cổng 8000 và Vite ở http://localhost:5173, tự tải lại khi sửa code
```

Vite chuyển tiếp các request `/api` sang cổng 8000. Có thể chạy riêng từng phần bằng `make api` và `make web`.

## Ghi chú

- `data/` và `models/` không được commit (xem `.gitignore`) — sẽ tự sinh khi chạy notebook.
- Nếu link tải dataset gốc bị thay đổi, xem hướng dẫn thay thế trong README của repo
  [`sonlam1102/vihsd`](https://github.com/sonlam1102/vihsd).
