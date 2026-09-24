# CHƯƠNG 4: KẾT LUẬN

## 4.1. Kết luận

Qua đề tài, nhóm đã tìm hiểu bài toán phát hiện bình luận độc hại, các hướng tiếp cận chính, và đặc biệt là các thành phần cốt lõi của kiến trúc Transformer Encoder gồm token embedding, positional encoding hình sin, scaled dot-product attention, multi-head attention, feed-forward network, residual connection và layer normalization. Từ phần lý thuyết này, nhóm tự cài đặt được toàn bộ Transformer Encoder bằng các khối cơ bản của PyTorch, không dùng các lớp Transformer có sẵn hay mô hình pretrained. Nhóm cũng xây dựng một pipeline hoàn chỉnh: tiền xử lý và tách từ tiếng Việt, xây dựng từ điển, huấn luyện với weighted cross-entropy và lịch học Noam, đánh giá và suy luận trên câu mới.

Trong lần chạy thực nghiệm chính thức trên dataset ViHSD, mô hình có 1.801.987 tham số, được huấn luyện 15 epoch trong khoảng 3 phút 35 giây trên Apple M1 Pro. Mô hình đạt macro-F1 0,6156 trên tập dev, **accuracy 0,8286 và macro-F1 0,6083 trên tập test**, phát hiện được 62,7% số bình luận HATE. Qua ba seed, test macro-F1 trung bình là 0,6069 ± 0,0080. Làm mốc tham chiếu, PhoBERT fine-tune trên cùng dữ liệu đạt test macro-F1 0,6539: cao hơn khoảng 0,047, nhưng lớn gấp khoảng 75 lần và cần thời gian huấn luyện gấp khoảng 20 lần.

Một kết quả quan trọng của đề tài là khảo sát ảnh hưởng của learning rate. Lần chạy ban đầu có lỗi trong cách kết hợp `LambdaLR` với lịch Noam, khiến learning rate thực tế chỉ khoảng $2{,}6 \times 10^{-6}$ và macro-F1 chỉ đạt 0,3780. Khi sửa thành learning rate quá lớn (khoảng $2{,}6 \times 10^{-3}$), kiến trúc Post-LN lại mất ổn định. Chỉ khi learning rate đỉnh ở khoảng $5{,}3 \times 10^{-4}$, mô hình mới hội tụ tốt. Thay Post-LN bằng Pre-LN cho kết quả tương đương qua ba seed (0,6029 ± 0,0072 so với 0,6069 ± 0,0080). Điều này cho thấy khi huấn luyện Transformer từ đầu, việc thiết lập đúng lịch học quan trọng không kém việc cài đặt đúng kiến trúc.

## 4.2. Hạn chế

- Lớp OFFENSIVE vẫn khó phân biệt (F1 = 0,3871), thường bị nhầm với cả CLEAN lẫn HATE; precision của HATE còn thấp (0,4546). Mô hình vẫn kém PhoBERT ở cả ba lớp.
- Mô hình được huấn luyện từ trọng số ngẫu nhiên, không có tri thức ngôn ngữ tiền huấn luyện, chỉ học từ khoảng 24 nghìn câu với phần lớn là CLEAN; mô hình bắt đầu overfitting từ khoảng epoch 7.
- Từ điển mức từ không xử lý được biến thể chính tả, teencode và từ mới; 8,4% token trên tập test là `<unk>`.
- Kết quả phụ thuộc vào chất lượng tách từ của underthesea, vốn được huấn luyện trên văn bản chuẩn, chưa tối ưu cho ngôn ngữ mạng xã hội.
- Kết quả dao động khoảng ±0,01 macro-F1 giữa các lần chạy (backend MPS không tất định tuyệt đối). Đề tài mới đánh giá ba seed cho hai cấu hình chính và khảo sát ba mức learning rate; chưa khảo sát các siêu tham số khác (số lớp, `d_model`, dropout, MAX_LEN). Vì đã xem kết quả test của nhiều cấu hình trong quá trình khảo sát, con số test có thể hơi lạc quan.
- PhoBERT chỉ được fine-tune một lần với một bộ siêu tham số, nên so sánh ở mục 3.4.5 chỉ mang tính tham chiếu.
- Mô hình chỉ xử lý theo dạng batch trong notebook, chưa đóng gói thành dịch vụ kiểm duyệt trực tuyến.

## 4.3. Hướng phát triển

- Tăng số seed (ví dụ 5–10) và dùng một tập validation riêng tách từ train để chọn siêu tham số, giữ tập test hoàn toàn độc lập cho lần đánh giá cuối cùng.
- Áp dụng early stopping theo dev macro-F1, tăng dropout hoặc weight decay để giảm overfitting; khảo sát thêm số lớp encoder, `d_model` và `MAX_LEN`.
- Khởi tạo lớp embedding bằng word embedding tiếng Việt đã huấn luyện sẵn (ví dụ fastText [15]) thay vì ngẫu nhiên, hoặc tự tiền huấn luyện mô hình bằng masked language modeling trên corpus bình luận không gán nhãn.
- Thay từ điển mức từ bằng tokenizer subword (BPE/WordPiece) để giảm tỉ lệ `<unk>` và xử lý tốt hơn các biến thể chính tả, teencode.
- Thử focal loss [14] hoặc tăng cường dữ liệu cho các lớp thiểu số để cải thiện cân bằng giữa precision và recall của OFFENSIVE và HATE.
- Trực quan hoá trọng số attention của mô hình đã huấn luyện trên các bình luận thật để phân tích mô hình "chú ý" vào những từ nào khi ra quyết định.
- Thu hẹp khoảng cách với PhoBERT [3] theo hướng tiền huấn luyện nhẹ: pretrain chính mô hình nhỏ này bằng masked language modeling trên corpus bình luận tiếng Việt, rồi fine-tune trên ViHSD.
