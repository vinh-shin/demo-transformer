# CHƯƠNG 1: MỞ ĐẦU

## 1.1. Lý do chọn đề tài

Mạng xã hội như Facebook, YouTube, TikTok đã trở thành không gian giao tiếp chính của hàng chục triệu người dùng Việt Nam. Bên cạnh những trao đổi tích cực, lượng bình luận mang tính xúc phạm, lăng mạ hoặc kích động thù ghét cũng tăng lên nhanh chóng. Các bình luận độc hại này ảnh hưởng tiêu cực đến tâm lý người đọc, làm xấu môi trường thảo luận chung và gây rủi ro pháp lý cho các nền tảng. Với khối lượng dữ liệu khổng lồ được tạo ra mỗi ngày, việc kiểm duyệt thủ công là không khả thi, đòi hỏi phải có các hệ thống tự động phát hiện bình luận độc hại.

Nhóm chúng em lựa chọn đề tài "Ứng dụng kỹ thuật Transformer trong bài toán phát hiện bình luận độc hại" xuất phát từ ba lý do chính sau.

- Thứ nhất, Transformer [1] là kiến trúc nền tảng của hầu hết các mô hình xử lý ngôn ngữ tự nhiên hiện đại như BERT [2], PhoBERT [3] hay các mô hình ngôn ngữ lớn. Việc tự cài đặt một Transformer Encoder hoàn chỉnh, dù ở quy mô nhỏ, là cách hiệu quả để nhóm nắm vững các khái niệm cốt lõi như self-attention, multi-head attention, positional encoding thay vì chỉ sử dụng các thư viện có sẵn như một "hộp đen".
- Thứ hai, bài toán phát hiện bình luận độc hại tiếng Việt có nhiều thách thức thực tế: ngôn ngữ mạng xã hội nhiều từ lóng, viết tắt, sai chính tả, teencode, emoji; dữ liệu mất cân bằng lớp nghiêm trọng khi phần lớn bình luận là bình thường. Đây là môi trường tốt để quan sát trung thực khả năng và giới hạn của một mô hình Transformer.
- Thứ ba, nhóm mong muốn làm việc với một bộ dữ liệu thực tế, có gán nhãn và được công bố khoa học là ViHSD [4], để kết quả thực nghiệm có thể tái lập và đối chiếu được.

Xuất phát từ các lý do trên, đề tài của nhóm chúng em tập trung xây dựng và phân tích một mô hình Transformer Encoder được **tự cài đặt từ đầu bằng PyTorch** và **huấn luyện từ trọng số ngẫu nhiên** (không dùng mô hình pretrained) cho bài toán phân loại bình luận thành ba lớp CLEAN, OFFENSIVE và HATE, đồng thời đánh giá trung thực những gì một Transformer không có tri thức ngôn ngữ tiền huấn luyện có thể đạt được trên dữ liệu tiếng Việt.

## 1.2. Mục tiêu đề tài

Đề tài hướng đến các mục tiêu cụ thể sau:

- Hệ thống hoá cơ sở lý thuyết về bài toán phát hiện bình luận độc hại và các hướng tiếp cận chính, trọng tâm là kiến trúc Transformer Encoder (embedding, positional encoding, scaled dot-product attention, multi-head attention, feed-forward, residual connection và layer normalization).
- Cài đặt một pipeline hoàn chỉnh từ dữ liệu thô: tải dữ liệu, làm sạch và tách từ tiếng Việt, xây dựng từ điển, mã hoá câu, xây dựng mô hình, huấn luyện, đánh giá và suy luận trên câu mới.
- Tự cài đặt toàn bộ các thành phần của Transformer Encoder bằng các khối cơ bản của PyTorch (`nn.Linear`, `nn.Embedding`, `nn.LayerNorm`,...), không dùng `nn.MultiheadAttention`, `nn.TransformerEncoder` hay thư viện HuggingFace `transformers`.
- Thực nghiệm trên dataset ViHSD, thu thập số liệu định lượng gồm số tham số mô hình, loss, accuracy, precision, recall, F1 theo từng lớp và macro-F1, confusion matrix; đánh giá trung thực hạn chế của mô hình huấn luyện từ đầu trên dữ liệu nhỏ và mất cân bằng.

## 1.3. Đối tượng và phạm vi nghiên cứu

Đối tượng nghiên cứu của đề tài là kiến trúc Transformer Encoder và ứng dụng của nó trong bài toán phân loại văn bản, cụ thể là phát hiện bình luận độc hại tiếng Việt trên mạng xã hội.

- Phạm vi lý thuyết: mô hình hoá bài toán phát hiện bình luận độc hại dưới dạng phân loại văn bản đa lớp; cấu trúc Transformer Encoder theo bài báo gốc *Attention Is All You Need* [1]; hàm mất mát cross-entropy có trọng số lớp, bộ tối ưu Adam, lịch học Noam và các độ đo đánh giá cho dữ liệu mất cân bằng.
- Phạm vi thực nghiệm: cài đặt và huấn luyện mô hình trên dataset ViHSD [4] gồm khoảng 33.400 bình luận, giữ nguyên cách chia train/dev/test của tác giả; mô hình dùng từ điển mức từ (word-level) tự xây từ tập train, trọng số khởi tạo ngẫu nhiên.
- Đề tài không triển khai: tiền huấn luyện (pretraining) trên corpus lớn, phần Decoder của Transformer, hay triển khai hệ thống kiểm duyệt trực tuyến. Mô hình pretrained PhoBERT [3] chỉ được fine-tune trên cùng dữ liệu để làm mốc tham chiếu, không phải đối tượng nghiên cứu chính.

## 1.4. Nội dung thực hiện

- Khảo sát tài liệu, bài báo khoa học về phát hiện ngôn từ thù ghét/xúc phạm, kiến trúc Transformer và dataset ViHSD.
- Xây dựng cơ sở lý thuyết, mô hình hoá bài toán phân loại bình luận ba lớp.
- Thiết kế pipeline xử lý: tiền xử lý văn bản (làm sạch, tách từ), xây dựng từ điển và mã hoá câu, Transformer Encoder, mean pooling và lớp phân loại, huấn luyện và đánh giá.
- Cài đặt bằng Python (PyTorch, underthesea, scikit-learn) trên Jupyter Notebook; thực nghiệm trên ViHSD; thu thập số liệu định lượng.
- Phân tích, đánh giá kết quả đạt được, hạn chế và đề xuất hướng phát triển.

## 1.5. Phương pháp thực hiện

Đề tài của nhóm chúng em kết hợp phương pháp tìm hiểu lý thuyết và phương pháp thực nghiệm xây dựng ứng dụng.

- Về mặt lý thuyết, nhóm tổng hợp và đọc một số công trình, bài báo khoa học đã công bố (chủ yếu là các bài báo open access trên arXiv và tại các hội nghị uy tín về xử lý ngôn ngữ tự nhiên và học máy) để xây dựng nền tảng lý luận nhất quán cho kiến trúc Transformer và bài toán phát hiện bình luận độc hại.
- Về mặt thực nghiệm, nhóm cài đặt trực tiếp từng thành phần của Transformer Encoder bằng PyTorch, huấn luyện trên dataset ViHSD, sau đó thu thập số liệu định lượng và đối chiếu với cơ sở lý thuyết đã trình bày.

## 1.6. Dự kiến kết quả

Sau khi hoàn thành, đề tài dự kiến đạt được các kết quả sau:

- Báo cáo trình bày có hệ thống cơ sở lý thuyết về bài toán phát hiện bình luận độc hại và kiến trúc Transformer Encoder, có trích dẫn nguồn tham khảo khoa học rõ ràng.
- Một notebook hoàn chỉnh, có thể chạy lại được, cài đặt Transformer Encoder từ đầu, huấn luyện và đánh giá trên ViHSD, kèm hàm suy luận cho câu bình luận mới.
- Số liệu kết quả thực nghiệm thực tế mà nhóm thực hiện được (không phải số liệu minh hoạ) bao gồm kích thước từ điển, số tham số mô hình, loss và macro-F1 theo từng epoch, classification report và confusion matrix trên tập test.
- Nhận xét, đánh giá trung thực về ưu điểm và hạn chế của một Transformer huấn luyện từ đầu không có tri thức tiền huấn luyện. Cuối cùng là đề xuất hướng phát triển tiếp theo.
