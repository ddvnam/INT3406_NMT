# Các Chỉ Số Đánh Giá Dịch Máy: BLEU, TER và METEOR

Tài liệu này mô tả cơ chế toán học và cách tính của ba chỉ số đánh giá phổ biến trong xử lý ngôn ngữ tự nhiên (NLP).

## 1. BLEU Score (Bilingual Evaluation Understudy)

**BLEU** là chỉ số phổ biến nhất, dựa trên độ chính xác (Precision) của các n-gram trùng khớp giữa bản dịch của máy (Candidate) và bản dịch tham chiếu (Reference).

### Cách tính toán

Quy trình tính BLEU gồm 3 thành phần chính:

#### A. Modified N-gram Precision ($p_n$)

Để tránh việc máy "ăn gian" bằng cách lặp lại một từ đúng nhiều lần, BLEU sử dụng "clipped counts".

1.  Đếm số lần xuất hiện của mỗi n-gram trong Candidate.
2.  Đếm số lần xuất hiện tối đa của n-gram đó trong bất kỳ câu Reference nào.
3.  Lấy giá trị nhỏ hơn giữa (1) và (2) làm tử số.

$$p_n = \frac{\sum_{C \in \{Candidates\}} \sum_{ngram \in C} Count_{clip}(ngram)}{\sum_{C' \in \{Candidates\}} \sum_{ngram' \in C'} Count(ngram')}$$

#### B. Brevity Penalty (BP)

BLEU là một metric thiên về Precision, nên nó có xu hướng ưu tiên các câu ngắn. BP được thêm vào để phạt các câu quá ngắn so với Reference.

Gọi $c$ là độ dài của Candidate và $r$ là độ dài của Reference:
$$BP = \begin{cases} 1 & \text{nếu } c > r \\ e^{(1 - r/c)} & \text{nếu } c \le r \end{cases}$$

#### C. Công thức tổng hợp

BLEU thường được tính trên tổ hợp của 1-gram đến 4-gram, sử dụng trung bình nhân (geometric mean):

$$BLEU = BP \times \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

_Trong đó:_

- $N$: Thường là 4.
- $w_n$: Trọng số cho mỗi loại n-gram (thường là $1/N$).

## 2. TER Score (Translation Edit Rate)

**TER** đo lường nỗ lực chỉnh sửa cần thiết để biến bản dịch của máy thành bản dịch tham chiếu. Điểm TER càng **thấp** thì chất lượng càng **tốt**.

### Cách tính toán

TER được tính bằng tỷ lệ số lượng thao tác chỉnh sửa trên tổng số từ của bản tham chiếu.

$$TER = \frac{\text{Số lượng thao tác chỉnh sửa (Edits)}}{\text{Độ dài trung bình của Reference}}$$

### Các thao tác chỉnh sửa

TER cho phép 4 loại thao tác (tương tự Levenshtein distance nhưng có thêm Shift):

1.  **Insertion (Chèn):** Thêm một từ vào.
2.  **Deletion (Xóa):** Xóa một từ đi.
3.  **Substitution (Thay thế):** Đổi từ sai thành từ đúng.
4.  **Shift (Dịch chuyển):** Di chuyển một chuỗi từ liền kề từ vị trí này sang vị trí khác (Đây là điểm đặc biệt giúp TER xử lý tốt sự khác biệt về trật tự từ).

## 3. METEOR Score (Metric for Evaluation of Translation with Explicit ORdering)

**METEOR** khắc phục điểm yếu của BLEU bằng cách tính đến cả Độ phủ (Recall) và sự tương đồng về ngữ nghĩa (từ đồng nghĩa, từ gốc). Nó thường tương quan tốt hơn với đánh giá của con người.

### Cách tính toán

Quy trình tính METEOR chia làm 3 giai đoạn:

#### A. Gióng hàng (Alignment)

Hệ thống cố gắng khớp từng từ trong Candidate với Reference theo thứ tự ưu tiên:

1.  **Exact:** Khớp chính xác từ.
2.  **Stem:** Khớp từ gốc (ví dụ: "running" khớp "run").
3.  **Synonym:** Khớp từ đồng nghĩa (dựa trên WordNet, ví dụ: "quick" khớp "fast").

#### B. Tính F-mean

METEOR tính Precision ($P$) và Recall ($R$) dựa trên các từ đã gióng hàng (unigrams). Tuy nhiên, METEOR ưu tiên Recall hơn rất nhiều:

$$F_{mean} = \frac{10PR}{R + 9P}$$

#### C. Tính Fragmentation Penalty (Phạt phân mảnh)

Để đánh giá trật tự từ, METEOR đếm số lượng "chunks" (các chuỗi từ liền kề đã được gióng hàng đúng thứ tự). Càng nhiều chunks rời rạc (phân mảnh) thì càng bị phạt nặng.

$$Penalty = 0.5 \times \left(\frac{\text{Số lượng chunks}}{\text{Số lượng unigrams đã khớp}}\right)^3$$

#### D. Công thức cuối cùng

$$METEOR = F_{mean} \times (1 - Penalty)$$
