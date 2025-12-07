import spacy
from typing import List
from underthesea import word_tokenize
from utils import clean_text

class Tokenizer:
    """
    Lớp bao đóng các thư viện tách từ (Spacy cho tiếng Anh, Underthesea cho tiếng Việt).
    """
    def __init__(self, language: str):
        """
        Khởi tạo tokenizer dựa trên ngôn ngữ.
        
        Args:
            language (str): Mã ngôn ngữ ('en' hoặc 'vi').
        """
        self.language = language
        self.nlp = None
        
        if language == 'en':
            try:
                self.nlp = spacy.load('en_core_web_sm')
            except OSError:
                print("Downloading en_core_web_sm...")
                from spacy.cli import download
                download('en_core_web_sm')
                self.nlp = spacy.load('en_core_web_sm')
        elif language == 'vi':
            # Underthesea không cần load model object nặng như spacy
            pass
        else:
            raise ValueError(f"Language '{language}' not supported yet.")

    def tokenize(self, text: str) -> List[str]:
        """
        Làm sạch và tách từ một câu văn bản.
        
        Args:
            text (str): Câu đầu vào.
            
        Returns:
            List[str]: Danh sách các token.
        """
        text = clean_text(text)
        
        if self.language == 'en':
            return [tok.text for tok in self.nlp.tokenizer(text)]
        elif self.language == 'vi':
            return word_tokenize(text, format="text").split()
        return text.split()

if __name__ == "__main__":
    en_tok = Tokenizer('en')
    vi_tok = Tokenizer('vi')
    
    print("EN:", en_tok.tokenize("Hello, World!"))
    print("VI:", vi_tok.tokenize("Xin chào, thế giới!"))