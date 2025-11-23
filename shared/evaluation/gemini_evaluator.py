from dotenv import load_dotenv
import os
import google.generativeai as genai
import json
import time
class GeminiEvaluator:
    def __init__(self, gemini_key = None):
        self.api_key = gemini_key

        if gemini_key:
            genai.configure(api_key=gemini_key)
            self.gemini_model = genai.GenerativeModel('gemini-2.5-pro')
        else:
            self.gemini_model = None
            print("Warning: Chưa có API Key cho Gemini, chức năng Gemini Score sẽ không hoạt động.")


    def compute_gemini_score_single_call(self, 
                                        sources: list[str], 
                                        predictions: list[str], 
                                        all_references: list[list[str]]) -> list[float]:
        """
        Sử dụng LLM (Gemini) để chấm điểm chất lượng bản dịch trong MỘT lần gọi API.
        
        Returns:
            list[float]: Danh sách các điểm số đã được trích xuất (0.0 đến 1.0).
        """
        if not self.gemini_model:
            print("Lỗi: Không có API Key.")
            return []

        if len(sources) != len(predictions) or len(predictions) != len(all_references):
            raise ValueError("Các danh sách Source, Prediction và Reference phải cùng kích thước.")

        # 1. Chuẩn hóa dữ liệu đầu vào thành cấu trúc JSON rõ ràng
        structured_input = []
        for s, p, refs in zip(sources, predictions, all_references):
            structured_input.append({
                "source": s,
                "predictions": p,
                "references": refs
            })

        prompt = f"""
        Bạn là chuyên gia đánh giá bản dịch. Nhiệm vụ của bạn là đánh giá TẤT CẢ các bản dịch trong danh sách sau.
        ĐÁNH GIÁ theo thang điểm 0.0 (rất tệ) đến 1.0 (hoàn hảo) dựa trên độ chính xác và độ trôi chảy.
        
        Đầu vào: 
        {json.dumps(structured_input, ensure_ascii=False, indent=2)}
        
        YÊU CẦU: Trả về MẢNG JSON duy nhất (list of objects) chứa {len(sources)} phần tử. 
        Mỗi phần tử chỉ chứa trường "score".
        
        Định dạng đầu ra BẮT BUỘC (Không thêm bất cứ văn bản nào khác):
        [
        {{"score": <float_score_1>}},
        {{"score": <float_score_2>}},
        ...
        ]
        """

        try:
            # 3. Gọi API
            response = self.gemini_model.generate_content(prompt)
            raw_json_text = response.text.strip()

            # 4. Xử lý và Phân tích JSON phức tạp
            
            # Làm sạch các markdown fence (```json) nếu có
            if raw_json_text.startswith("```"):
                raw_json_text = raw_json_text.strip('`').strip('json').strip()
            
            # Phân tích chuỗi JSON lớn thành Python list
            json_array = json.loads(raw_json_text)
            
            # 5. Trích xuất Score
            scores = [float(item.get("score", 0.0)) for item in json_array]
            
            return scores
            
        except json.JSONDecodeError:
            print("ERROR: Không thể phân tích cú pháp JSON lớn. Có thể mô hình đã trả về JSON không hợp lệ hoặc bị cắt bớt.")
            return [0.0] * len(sources)
            
        except Exception as e:
            print(f"ERROR API: Lỗi trong quá trình gọi hoặc xử lý: {e}")
            return [0.0] * len(sources)

def main():
    sources = ["con mèo đang ở trên bàn", "xin chào thế giới"]
    preds = [
        "the cat is on a lebron james",
        "hello world"
    ]
    
    targets = [
        ["the cat is on the table", "there is a cat on the table"], # 2 refs cho câu 1
        ["hello world"]                                         # 1 ref cho câu 2
    ]
    load_dotenv(override=True)
    gemini_key = os.getenv("gemini_api_key")
    gemini = GeminiEvaluator(gemini_key=gemini_key)
    evaluation = gemini.compute_gemini_score_single_call(sources=sources, predictions=preds, all_references=targets)
    print(evaluation)

