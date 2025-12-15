# Tổng Quan Các Hướng Tiếp Cận Xây Dựng LLM Cho Lĩnh Vực Y Tế

Có 3 hướng tiếp cận chính để xây dựng một Mô hình Ngôn ngữ Lớn (LLM) phục vụ chuyên biệt cho lĩnh vực y học:

---

## 1. Pre-training (Huấn luyện trước)

**Định nghĩa:** Sử dụng một cấu trúc mô hình với bộ tham số ban đầu ngẫu nhiên (chưa học gì) và dùng một bộ dữ liệu khổng lồ để huấn luyện từ đầu (from scratch).

**Các hướng tiếp cận để huấn luyện:**

* **Masked Language Modeling (MLM):** Đưa ra nhiệm vụ cho mô hình với đầu vào là một câu, trong đó một phần của câu bị xóa (che) đi. Mô hình phải dùng ngữ cảnh để dự đoán phần bị thiếu, sử dụng chính toàn bộ câu gốc làm nhãn (label).
* **Next Sentence Prediction (NSP):** Dự đoán xem câu tiếp theo có phải là câu nối tiếp hợp lý của câu trước đó hay không.
* **Next Token Prediction (NTP):** Dự đoán từ (token) tiếp theo trong chuỗi dựa trên các từ trước đó.

---

## 2. Fine-tuning (Tinh chỉnh)

**Định nghĩa:** Do việc train model từ đầu rất tốn kém, phương pháp này sử dụng những LLM tổng quát (General LLMs) đã có sẵn để tinh chỉnh lại (finetune) dựa vào lượng dữ liệu chuyên biệt hóa cho lĩnh vực y tế.

### Các hướng xây dựng bộ dữ liệu (Dataset) cho Fine-tuning:

#### A. SFT (Supervised Fine-Tuning)
* **Cấu trúc tổng quát:** Dạng `1 câu hỏi - 1 câu trả lời`.
* **Nguồn dữ liệu:**
    * Đoạn hội thoại giữa bệnh nhân và bác sĩ.
    * Bộ câu hỏi trắc nghiệm hoặc tự luận.
    * Đồ thị kiến thức (Knowledge Graphs) về y học.

#### B. IFT (Instruction Fine-Tuning)
* **Cấu trúc tổng quát:** `1 câu chỉ dẫn - 1 đoạn nội dung - 1 câu trả lời`.
    * *Câu chỉ dẫn:* Giúp mô hình hiểu cần làm gì với nội dung đầu vào để ra được kết quả mong muốn.
* **Ví dụ:**
    > **Chỉ dẫn:** Dịch câu này sang tiếng Việt chuyên ngành y.
    > **Nội dung:** (Nội dung đoạn văn bản y khoa tiếng Anh).
    > **Output:** (Đoạn văn bản đã được dịch).
* **Ưu điểm:** Giúp mô hình linh hoạt hơn trong nhiều tác vụ khác nhau thay vì chỉ làm một công việc cố định như SFT.

### PEFT (Parameter-Efficient Fine-Tuning)
Hướng tiếp cận giúp việc fine-tuning đỡ tốn kém tài nguyên hơn bằng cách không huấn luyện lại toàn bộ tham số.

1.  **LoRA (Low-Rank Adaptation):** Đóng băng toàn bộ tham số (parameters) gốc của mô hình. Thêm vào các module Self-Attention những ma trận nhỏ có thể train được để tối ưu hóa cho lĩnh vực chuyên biệt.
2.  **Prefix Tuning:** Thêm những vector gọi là "vector ngữ cảnh" vào các vector đầu vào của từng lớp Transformer, giúp mô hình hiểu được ngữ cảnh đặc thù khi fine-tuning.
3.  **Adapter:** Thêm những module Neural Networks (NN) nhỏ vào bên trong các lớp Transformer để train. Các tham số còn lại của LLM được giữ nguyên. Độ lớn của các module NN này chỉ cần đủ để học kiến thức lĩnh vực mới.

---

## 3. Prompting (Kỹ thuật Lời nhắc)

**Định nghĩa:** Sử dụng thuần túy câu lệnh (prompt) để khai thác mô hình. Cách này giả định mô hình tổng quát đã có sẵn đầy đủ kiến thức về lĩnh vực đó.

**Các hướng tiếp cận:**

* **ICL (In-Context Learning):** Trước khi đưa ra nhiệm vụ, cần cung cấp cho mô hình:
    * Yêu cầu cụ thể và mục đích.
    * Ngữ cảnh của yêu cầu.
    * Một vài ví dụ mẫu (Few-shot) để mô hình hiểu nhiệm vụ.
* **CoT (Chain of Thought):** Yêu cầu mô hình diễn giải từng bước suy luận (trung gian) thay vì đưa ra ngay câu trả lời cuối cùng.
    * *Lợi ích:* Tăng độ tin cậy và tính giải thích được (explainability).
* **Prompt Tuning:** Thay vì train các tham số kiến thức, ta train các tham số liên quan đến quá trình **Embedding** (cách prompt được chuyển hóa thành tín hiệu mà mô hình hiểu).

### RAG (Retrieval-Augmented Generation)
Phương pháp hỗ trợ đặc biệt hữu hiệu cho các lĩnh vực kiến thức cập nhật liên tục (như y tế), giúp tránh hallucination và kiến thức lỗi thời (out-dated).

* **Cơ chế hoạt động:**
    1.  **Retrieval (Truy xuất):** Nhận prompt đầu vào, tìm kiếm thông tin liên quan từ kho kiến thức bên ngoài (Internet, CSDL y khoa).
    2.  **Augmentation (Tăng cường):** Kết hợp kiến thức vừa tìm được vào câu prompt gốc để cung cấp thêm ngữ cảnh (context).
    3.  **Generation (Tạo sinh):** Chuyển bộ prompt đã được làm giàu vào mô hình để sinh câu trả lời.

---

## 4. Ứng Dụng Dịch Ngôn Ngữ Trong Y Tế

### Một số Model tiêu biểu:
*(Phần này có thể bổ sung các model như: MedPaLM, BioBERT, ClinicalBERT, VinAI-Translate, v.v.)*

### Lưu ý khi Fine-tuning cho tác vụ dịch thuật:
1.  **Dữ liệu:** Cần tìm những dataset uy tín, đã được kiểm định chất lượng trước đó.
2.  **Đánh giá:** Các thang đánh giá ngôn ngữ thông thường (như BLEU, ROUGE) cần được bổ sung bằng các tiêu chuẩn đánh giá chuyên môn y tế.
3.  **Hỗ trợ:** Có thể kết hợp **RAG** để tra cứu các thuật ngữ khó/mới từ bên ngoài làm kiến thức nền cho mô hình trước khi thực hiện dịch.

---

## 5. Thách Thức và Giải Pháp: Hallucination (Ảo giác)

Hiện tượng mô hình tự sinh ra thông tin sai lệch hoặc bịa đặt.

### Các giải pháp khắc phục:

#### A. Trong quá trình Huấn luyện (Training-time correction)
Điều chỉnh trực tiếp trọng số (weights) để giảm xác suất sinh ra ảo giác.
* **Học tăng cường nhất quán thực tế (Factually consistent reinforcement learning):** Thưởng (reward) cho các câu trả lời đúng sự thật.
* **Học tương phản (Contrastive learning):** Huấn luyện để mô hình phân biệt rõ ràng giữa thông tin đúng và sai.

#### B. Trong quá trình Sinh ra (Generation-time correction)
* **Rút nhiều mẫu (Drawing multiple samples):** Sinh ra nhiều câu trả lời, sau đó chọn câu có điểm số tin cậy (confidence score) cao nhất.
* **Sử dụng điểm tin cậy:** Xác định và loại bỏ ảo giác dựa trên ngưỡng tin cậy trước khi xuất kết quả.

#### C. Trong quá trình Tăng cường Prompt (Retrieval-augmented correction)
* **Sử dụng tài liệu thực tế:** Truy xuất tài liệu liên quan chèn vào prompt để làm cơ sở tham chiếu.
* **Chuỗi truy xuất (Chain-of-retrieval):** Thực hiện nhiều bước truy xuất để xác minh thông tin (tương tự việc kiểm tra chéo nhiều nguồn).

### Vấn đề về Thang đánh giá
Các thang đánh giá hiện tại thường thiếu sót trong việc đo lường:
* **Trustworthiness:** Độ tin cậy.
* **Faithfulness:** Tính trung thực (bám sát ngữ cảnh nguồn).