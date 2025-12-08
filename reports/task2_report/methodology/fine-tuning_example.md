# Báo Cáo Tóm Tắt: Cải Thiện Dịch Máy Y Tế Việt - Anh (Improving VE Translation)

**Mục tiêu chính:** Bài báo đề xuất bộ dữ liệu **MedEV** để tinh chỉnh (fine-tune) các mô hình dịch máy và thực hiện đánh giá so sánh hiệu suất giữa các mô hình nổi tiếng trước và sau khi fine-tune trên bộ dữ liệu này.

---

## 1. Kết Quả Thực Nghiệm

Kết quả so sánh hiệu suất giữa các mô hình được chia thành hai giai đoạn:

* **Trước khi Fine-tune (Pre-fine-tuning):**
    * **Google Translate:** Đạt điểm số (score) cao nhất trong các mô hình/công cụ khi chưa được huấn luyện lại trên tập dữ liệu chuyên biệt.

* **Sau khi Fine-tune (Post-fine-tuning):**
    * **vinai-translate:** Là mô hình có sự cải thiện điểm số rõ rệt nhất và đạt hiệu suất cao nhất trong số các mô hình được thử nghiệm tinh chỉnh.

---

## 2. Chi Tiết Quá Trình Fine-tune (Implementation Details)

Dưới đây là thông số kỹ thuật và quy trình thực hiện tinh chỉnh cho các mô hình dịch máy (Lưu ý: Không áp dụng cho ChatGPT do sử dụng cơ chế In-Context Learning).

### A. Các mô hình được chọn
Quá trình này áp dụng cho 4 mô hình nền tảng:
* `vinai-translate`
* `envit5-translation`
* `mBART`
* `envit5-base`

### B. Cấu hình và Tham số Huấn luyện
* **Dữ liệu:** Sử dụng bộ dữ liệu **MedEV** (xây dựng bởi nhóm tác giả).
* **Thư viện:** `transformers` của HuggingFace.
* **Số vòng lặp (Epochs):** 5 epochs.
* **Tối ưu hóa (Optimizer):** AdamW.
* **Tốc độ học (Learning Rate):** Khởi tạo ở mức `5e-5`.
* **Độ dài chuỗi (Sequence Length):** Tối đa 256 token.
* **Bước khởi động (Warm-up steps):** 1250 bước.

### C. Phần cứng và Hiệu suất
* **GPU:** 4x NVIDIA A100.
* **Chế độ chính xác:** Sử dụng huấn luyện chính xác hỗn hợp (**Mixed Precision Training - fp16**) để tiết kiệm bộ nhớ và tăng tốc độ.
* **Batch size:**
    * 4 trên mỗi GPU.
    * Sử dụng **Tích lũy gradient (Gradient Accumulation)** qua 8 bước (giúp mô phỏng batch size lớn hơn giới hạn phần cứng).

### D. Quy trình Đánh giá và Chọn Mô hình (Checkpoint Selection)
* **Giải mã (Decoding):** Sử dụng kỹ thuật **Beam Search** với kích thước beam (`beam size`) là 5.
* **Tần suất đánh giá:** Kiểm tra sau mỗi **1000 bước** huấn luyện.
* **Tiêu chí chọn Best Model:** Checkpoint đạt điểm **BLEU cao nhất trên tập Validation** sẽ được chọn để đánh giá chính thức trên tập Test.
* **Thang đo:**
    * BLEU (SacreBLEU, có phân biệt hoa thường - *case-sensitive*).
    * TER.
    * METEOR.

---

## 3. Đánh Giá Các Mô Hình LLM (Không Fine-tune)

Đối với các mô hình ngôn ngữ lớn (LLM) không thể hoặc không thực hiện fine-tune (như ChatGPT), bài báo áp dụng phương pháp sau:

### Sử dụng Prompting
* Bài báo cung cấp các **lệnh prompt mẫu** để đánh giá điểm số của các mô hình phổ biến.
* Mục đích: Tạo cơ sở so sánh (baseline) giữa các mô hình LLM đại chúng với mô hình chuyên biệt được fine-tune của nhóm tác giả.

### Xử lý Output (Hậu xử lý)
* **Vấn đề:** Các LLM thường sinh ra các câu thừa, lời dẫn dắt hoặc giải thích nằm ngoài nội dung dịch thuật chính.
* **Giải pháp:** Cần thực hiện các bước loại bỏ các nội dung thừa này, chỉ giữ lại phần bản dịch lõi trước khi đưa vào tính toán điểm số đánh giá.