# Báo Cáo Tổng Quan: COMET-M - Metric Đánh Giá Dịch Máy Chuyên Biệt Cho Y Tế

## 1. Tổng Quan Giải Pháp Đề Xuất

**Giải pháp:**
Nhóm tác giả đề xuất **COMET-M**, một công cụ đánh giá dịch máy dựa trên mạng nơ-ron (Neural Metric), được thiết kế và tối ưu hóa chuyên biệt cho lĩnh vực y tế.

**Mục tiêu:**
Tạo ra một công cụ đánh giá tự động có **độ tương quan cao** với các đánh giá chất lượng từ chuyên gia y tế (con người), khắc phục hạn chế của các thang đo truyền thống.

---

## 2. Xây Dựng Dữ Liệu: CorpMMT (Data Construction)

### Vấn đề (Problem)
* **Thiếu hụt dữ liệu:** Lĩnh vực đánh giá dịch máy thiếu các bộ dữ liệu chất lượng cao để huấn luyện các metric chuyên ngành. Các bộ dữ liệu hiện có chủ yếu là tin tức hoặc văn bản phổ thông.
* **Thiếu nhãn đánh giá sâu:** Không có bộ dữ liệu nào cung cấp nhãn đánh giá chi tiết về mức độ nghiêm trọng của các lỗi sai y khoa (MQM scores).

### Giải pháp (Solution)
* **Giới thiệu CorpMMT:** Tác giả xây dựng **CorpMMT**, một tập dữ liệu đa ngữ (Anh-Đức, Anh-Trung, v.v.) bao gồm các văn bản chuyên ngành y tế.
* **Quy trình chú giải:**
    * Dữ liệu được chú giải thủ công bởi các chuyên gia dựa trên khung **MQM (Multidimensional Quality Metrics)**.
    * Các lỗi được phân loại kỹ lưỡng (ví dụ: sai thuật ngữ, sai ý nghĩa, độ trôi chảy).
    * **Gán trọng số phạt:** Các lỗi được phân cấp mức độ nghiêm trọng: **Minor** (Nhẹ), **Major** (Lớn), **Critical** (Nghiêm trọng).
* **Kết quả:** Việc này tạo ra **"Ground Truth"** (chân lý) để dạy cho mô hình phân biệt được thế nào là một bản dịch y tế tồi và nguy hiểm.

---

## 3. Phương Pháp Luận: COMET-M (Methodology)

### Vấn đề (Problem)
Mô hình COMET gốc (được huấn luyện trên dữ liệu WMT chung) **thiếu kiến thức miền (domain knowledge)** về y tế.
* *Ví dụ:* Mô hình gốc không hiểu rằng việc dịch sai "cao huyết áp" thành "huyết áp thấp" là một lỗi **nghiêm trọng** (ảnh hưởng tính mạng), trong khi nó có thể chỉ coi đây là một lỗi từ vựng thông thường.

### Giải pháp (Solution)
* **Kiến trúc nền tảng:** Sử dụng kiến trúc **COMET** (dựa trên XLM-RoBERTa) làm mô hình tiền huấn luyện (Pre-trained model).
* **Thích ứng miền (Domain Adaptation):** Thực hiện tinh chỉnh (Fine-tune) mô hình COMET gốc bằng tập dữ liệu **CorpMMT** vừa xây dựng.
* **Cơ chế học:** Mô hình học cách **dự đoán điểm số chất lượng (MQM score)** thay vì chỉ nhìn vào sự tương đồng chuỗi ký tự bề mặt. Điều này giúp COMET-M "học" được tư duy và tiêu chuẩn đánh giá khắt khe của một chuyên gia y tế.

---

## 4. Thực Nghiệm & Kết Quả (Experiments & Results)

### Vấn đề (Problem)
Cần chứng minh tính hiệu quả vượt trội của COMET-M so với các metric "quốc dân" phổ biến hiện nay như **BLEU** hay **COMET-22** trong môi trường y tế đặc thù.

### Giải pháp & Kết quả
* **Phương pháp đánh giá:** Nhóm tác giả đánh giá sự tương quan (**Correlation**) giữa điểm số do metric chấm và điểm số do con người chấm (sử dụng chỉ số **Kendall’s Tau**).
* **Kết quả chính:**
    * **Độ tương quan cao:** COMET-M đạt độ tương quan cao hơn đáng kể so với BLEU, CHRF và cả mô hình COMET gốc trên các cặp ngôn ngữ thử nghiệm.
    * **Nhận diện lỗi nghiêm trọng:** COMET-M nhận diện tốt hơn các bản dịch chứa **lỗi y khoa nghiêm trọng (Critical errors)** và phạt nặng các bản dịch này. Đây là điểm vượt trội so với BLEU, vốn thường bỏ qua mức độ nguy hiểm của lỗi sai nghĩa trong y văn.