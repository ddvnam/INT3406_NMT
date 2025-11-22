import torch
from torchmetrics.text import BLEUScore, TranslationEditRate
import os
from nltk.translate.meteor_score import meteor_score
import statistics

class TranslationEvaluator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Khởi tạo các metrics đánh giá trên device (CPU/GPU).
        """
        self.device = device
        
        # 1. Khởi tạo các metric từ TorchMetrics
        # n_gram=4 là chuẩn cho BLEU-4
        self.bleu_metric = BLEUScore(n_gram=4).to(device)
        
        # TER (càng thấp càng tốt)
        self.ter_metric = TranslationEditRate().to(device)
        
        self.meteor_metric = {}    

    def calculate_corpus_meteor_nltk(self, predictions: list[str], all_references: list[list[str]]) -> float:
        """
        Tính điểm METEOR cho toàn bộ tập dữ liệu (Corpus).
        
        Args:
            predictions (list[str]): Danh sách các câu dịch của model (Hypothesis).
            all_references (list[list[str]]): Danh sách các Reference tương ứng. 
                                            Mỗi phần tử trong list ngoài là list các refs cho 1 câu.
                                            Ví dụ: [['ref1_c1', 'ref2_c1'], ['ref1_c2']]
                                            
        Returns:
            float: Điểm METEOR trung bình của Corpus (thang 0.0 đến 1.0).
        """
        if len(predictions) != len(all_references):
            raise ValueError("Số lượng câu dự đoán và danh sách References phải bằng nhau.")

        corpus_scores = []

        for pred_sent, refs_for_sent in zip(predictions, all_references):
            pred_tokens = pred_sent.split()
            ref_tokens_list = [ref.split() for ref in refs_for_sent]
            
            sentence_scores = []
            for ref_tokens in ref_tokens_list:
                score = meteor_score([ref_tokens], pred_tokens)
                sentence_scores.append(score)
                
            # Lấy điểm METEOR tốt nhất (Max) cho câu hiện tại
            best_sentence_score = max(sentence_scores) if sentence_scores else 0.0
            
            corpus_scores.append(best_sentence_score)

        # 4. Tính trung bình cộng của tất cả các điểm METEOR của từng câu
        if not corpus_scores:
            return 0.0
            
        final_corpus_meteor = statistics.mean(corpus_scores)
        
        return final_corpus_meteor

    def compute_traditional_metrics(self, preds: list[str], target: list[list[str]]):
        """
        Tính toán BLEU, TER, METEOR cùng lúc.
        
        Args:
            preds: List các câu model dịch. VD: ["hello world", "cat sits"]
            target: List các list tham chiếu. VD: [["hello world"], ["the cat sits"]]
                    (Lưu ý: TorchMetrics yêu cầu target phải là List[List[str]])
        
        Returns:
            Dictionary chứa kết quả.
        """
        # Update state cho metrics
        self.bleu_metric.update(preds, target)
        self.ter_metric.update(preds, target)
        self.meteor_metric = self.calculate_corpus_meteor_nltk(predictions=preds, all_references=target)
        # Tính toán (Compute)
        # Torchmetrics trả về Tensor, ta chuyển về float python để dễ đọc
        results = {
            "BLEU": self.bleu_metric.compute().item(),
            "TER": self.ter_metric.compute().item(),
            "METEOR": self.meteor_metric
        }
        
        # Reset state để dùng cho lần sau (quan trọng!)
        self.bleu_metric.reset()
        self.ter_metric.reset()

        return results
