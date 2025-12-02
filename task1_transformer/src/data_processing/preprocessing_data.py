import re
import spacy
from torchtext import data
from underthesea import word_tokenize

class Tokenizer:
    def __init__(self, language):
        if language == 'en':
            self.nlp = spacy.load('en_core_web_sm')
        elif language == 'vi':
            self.nlp = None
        self.language = language

    def tokenize(self, text):
        if self.language == 'en':
            return [tok.text for tok in self.nlp.tokenizer(text)]
        elif self.language == 'vi':
            return word_tokenize(text, format="text").split()

    def clean_text(self, text):
        """Chuẩn hóa văn bản: lowercase, xóa ký tự đặc biệt, xử lý khoảng trắng."""
        text = str(text).lower()
        # Loại bỏ các ký tự đặc biệt giữ lại dấu câu cơ bản
        text = re.sub(r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", text)
        text = re.sub(r"[ ]+", " ", text)
        text = re.sub(r"\!+", "!", text)
        text = re.sub(r"\,+", ",", text)
        text = re.sub(r"\?+", "?", text)
        return text.strip()
    
def create_fields(src_lang='en', tar_lang='vi'):
    """
    Tạo các Field của TorchText để định nghĩa cách xử lý dữ liệu.
    
    Args:
        src_lang (str): Mã ngôn ngữ nguồn (ví dụ: 'en').
        trg_lang (str): Mã ngôn ngữ đích (ví dụ: 'vi').
        
    Returns:
        tuple: (SRC Field, TRG Field)
    """
    src_tokenizer = Tokenizer(src_lang)
    tar_tokenizer = Tokenizer(tar_lang)

    SRC = data.Field(
        tokenize=src_tokenizer.tokenize,
        lower=True
    )
    TAR = data.Field(
        tokenize=tar_tokenizer.tokenize,
        lower=True,
        init_token='<sos>',
        eos_token='<eos>'
    )
    return SRC, TAR

