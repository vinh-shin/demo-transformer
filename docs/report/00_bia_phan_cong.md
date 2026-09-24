<div align="center">

**BỘ GIÁO DỤC VÀ ĐÀO TẠO**
**TRƯỜNG ĐẠI HỌC CÔNG NGHỆ KỸ THUẬT TP.HCM**
KHOA **CÔNG NGHỆ THÔNG TIN**

[LOGO HCMUTE]

# BÁO CÁO ĐỒ ÁN MÔN HỌC
# [TÊN MÔN HỌC]

**ĐỀ TÀI:**
**ỨNG DỤNG KỸ THUẬT TRANSFORMER**
**TRONG BÀI TOÁN PHÁT HIỆN BÌNH LUẬN ĐỘC HẠI**

</div>

> **GVHD:** PGS.TS. Hoàng Văn Dũng
> **Học viên thực hiện:**
> 2611318 - **Vòng Vĩnh Shìn**
> 2611302 - **Trần Nhật Duật**
> [MSHV] - **Hồ Đức Thắng**

<div align="center">

***TP. Hồ Chí Minh**, tháng [..] năm 2026*

</div>

---

## THÔNG TIN NHÓM VÀ TỶ LỆ ĐÓNG GÓP

| STT | MSHV | Họ và tên | Tỷ lệ đóng góp (%) | Ghi chú |
|---|---|---|---|---|
| 1 | 2611318 | Vòng Vĩnh Shìn | 100% | |
| 2 | 2611302 | Trần Nhật Duật | 100% | |
| 3 | [MSHV] | Hồ Đức Thắng | 100% | |

## BẢNG PHÂN CÔNG NHIỆM VỤ

> [TODO: nhóm chỉnh lại phân công cho đúng thực tế. Dưới đây là bản đề xuất.]

| STT | Họ và tên | Nhiệm vụ được phân công | Kết quả |
|---|---|---|---|
| 1 | Trần Nhật Duật | Tìm hiểu bài toán phát hiện bình luận độc hại, các hướng tiếp cận (luật, học máy truyền thống, RNN/CNN), khảo sát dataset ViHSD; biên soạn phần lý thuyết liên quan. | Hoàn thành |
| 2 | Vòng Vĩnh Shìn | Cài đặt khối tiền xử lý dữ liệu (làm sạch, tách từ), xây dựng vocab, mã hoá câu, `Dataset`/`DataLoader`; tham gia thực nghiệm. | Hoàn thành |
| 3 | Vòng Vĩnh Shìn | Nghiên cứu kiến trúc Transformer (self-attention, multi-head attention, positional encoding, residual + LayerNorm), biên soạn phần lý thuyết liên quan. | Hoàn thành |
| 4 | Trần Nhật Duật | Cài đặt Transformer Encoder từ đầu bằng PyTorch, vòng lặp huấn luyện với Noam schedule, đánh giá trên tập test; phân tích kết quả và lỗi dự đoán. | Hoàn thành |
| 5 | Hồ Đức Thắng | Khảo sát ảnh hưởng của learning rate (phát hiện lỗi kết hợp `LambdaLR` với lịch Noam) và vị trí LayerNorm (Post-LN/Pre-LN), chạy thực nghiệm nhiều seed và tổng hợp kết quả (mục 3.4.4). | Hoàn thành |
| 6 | Hồ Đức Thắng | Fine-tune mô hình pretrained PhoBERT trên cùng dữ liệu làm mốc tham chiếu, đối chiếu với mô hình tự cài đặt (mục 3.4.5). | Hoàn thành |

---

## MỤC LỤC

- CHƯƠNG 1: MỞ ĐẦU
  - 1.1. Lý do chọn đề tài
  - 1.2. Mục tiêu đề tài
  - 1.3. Đối tượng và phạm vi nghiên cứu
  - 1.4. Nội dung thực hiện
  - 1.5. Phương pháp thực hiện
  - 1.6. Dự kiến kết quả
- CHƯƠNG 2: CƠ SỞ LÝ THUYẾT
  - 2.1. Tổng quan về bài toán phát hiện bình luận độc hại
  - 2.2. Phân loại các phương pháp phát hiện bình luận độc hại
    - 2.2.1. Phương pháp dựa trên luật và từ điển
    - 2.2.2. Phương pháp học máy truyền thống
    - 2.2.3. Phương pháp học sâu tuần tự (RNN/LSTM, CNN)
    - 2.2.4. So sánh mô hình tuần tự và Transformer
  - 2.3. Kiến trúc Transformer Encoder
    - 2.3.1. Token Embedding
    - 2.3.2. Positional Encoding
    - 2.3.3. Scaled Dot-Product Attention và Multi-Head Attention
    - 2.3.4. Position-wise Feed-Forward Network
    - 2.3.5. Residual Connection và Layer Normalization
  - 2.4. Huấn luyện và đánh giá mô hình phân loại
    - 2.4.1. Hàm mất mát có trọng số lớp
    - 2.4.2. Bộ tối ưu Adam và lịch học Noam
    - 2.4.3. Các độ đo đánh giá
- CHƯƠNG 3: GIẢI PHÁP
  - 3.1. Một số kiến trúc tham khảo
  - 3.2. Sơ đồ tổng quát giải pháp
    - 3.2.1. Khối tiếp nhận và tiền xử lý dữ liệu
    - 3.2.2. Khối xây dựng từ điển và mã hoá câu
    - 3.2.3. Khối Embedding và Positional Encoding
    - 3.2.4. Khối Transformer Encoder
    - 3.2.5. Khối Mean Pooling và phân loại
    - 3.2.6. Khối huấn luyện, đánh giá và suy luận
  - 3.3. Thư viện sử dụng và môi trường cài đặt
    - 3.3.1. Thư viện sử dụng
    - 3.3.2. Môi trường cài đặt
  - 3.4. Thực nghiệm
    - 3.4.1. Môi trường và thiết lập thực nghiệm
    - 3.4.2. Quá trình thực hiện qua từng bước
    - 3.4.3. Kết quả thực nghiệm
    - 3.4.4. Khảo sát ảnh hưởng của learning rate và vị trí LayerNorm
    - 3.4.5. Đối chiếu với mô hình pretrained PhoBERT
  - 3.5. Phân tích và đánh giá kết quả
- CHƯƠNG 4: KẾT LUẬN
  - 4.1. Kết luận
  - 4.2. Hạn chế
  - 4.3. Hướng phát triển
- TÀI LIỆU THAM KHẢO

## DANH MỤC BẢNG

- Bảng 1: So sánh mô hình tuần tự (RNN/LSTM) và Transformer
- Bảng 2: So sánh một số kiến trúc phát hiện bình luận độc hại và giải pháp trong đề tài
- Bảng 3: Danh sách thư viện sử dụng
- Bảng 4: Phân bố nhãn của dataset ViHSD
- Bảng 5: Thông số cấu hình thực nghiệm
- Bảng 6: Kết quả huấn luyện trên tập dev theo epoch
- Bảng 7: Kết quả phân loại chi tiết trên tập test
- Bảng 8: Kết quả định lượng thực nghiệm
- Bảng 9: Kết quả khảo sát learning rate và vị trí LayerNorm
- Bảng 10: So sánh Post-LN và Pre-LN qua 3 seed
- Bảng 11: So sánh Transformer tự cài đặt với PhoBERT fine-tune

## DANH MỤC HÌNH ẢNH

- Hình 1: Kiến trúc một lớp Transformer Encoder (Post-LN)
- Hình 2: Cơ chế Multi-Head Self-Attention
- Hình 3: Ma trận trọng số self-attention trên câu ví dụ "Tôi yêu học máy"
- Hình 4: Sơ đồ tổng quát pipeline phát hiện bình luận độc hại
- Hình 5: Phân bố độ dài câu (số token) sau tách từ trên tập train
- Hình 6: Train loss, accuracy và macro-F1 trên tập dev theo epoch
- Hình 7: Confusion matrix trên tập test (số lượng và chuẩn hoá theo hàng)
- Hình 8: Train loss và dev macro-F1 theo epoch của bốn cấu hình huấn luyện
