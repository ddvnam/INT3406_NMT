import torch
import os
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from typing import List, Tuple, Optional

from load_data import read_data
from preprocessing_data import Tokenizer

SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'

def seed_everything(seed=42):
    """
    Cố định seed cho các thư viện ngẫu nhiên để đảm bảo kết quả tái lập được (reproducibility).
    
    Args:
        seed (int): Giá trị seed cần thiết lập. Mặc định là 42.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class Vocabulary:
    """
    Lớp quản lý từ điển (Vocabulary), chịu trách nhiệm ánh xạ giữa token (chuỗi ký tự) và index (số nguyên).
    
    Attributes:
        stoi (Dict[str, int]): Ánh xạ từ Token sang Index (String to Int).
        itos (Dict[int, str]): Ánh xạ từ Index sang Token (Int to String).
    """
    def __init__(self, tokens: List[str] = None, min_freq: int = 2):
        """
        Khởi tạo Vocabulary từ danh sách các token.
        
        Args:
            tokens (List[str], optional): Danh sách tất cả các token từ tập dữ liệu huấn luyện để xây dựng từ điển.
                Nếu là None, chỉ khởi tạo các token đặc biệt. Mặc định là None.
            min_freq (int): Tần suất xuất hiện tối thiểu để một token được thêm vào từ điển.
                Các token xuất hiện ít hơn ngưỡng này sẽ được coi là <unk>. Mặc định là 2.
        """
        self.stoi = {PAD_TOKEN: 0, UNK_TOKEN: 1, SOS_TOKEN: 2, EOS_TOKEN: 3}
        self.itos = {0: PAD_TOKEN, 1: UNK_TOKEN, 2: SOS_TOKEN, 3: EOS_TOKEN}
        
        if tokens:
            counter = Counter(tokens)
            idx = 4
            for token, freq in counter.items():
                if freq >= min_freq:
                    self.stoi[token] = idx
                    self.itos[idx] = token
                    idx += 1
                
    def __len__(self):
        """Trả về kích thước hiện tại của từ điển (số lượng token)."""
        return len(self.stoi)
        
    def numericalize(self, text_tokens: List[str]) -> List[int]:
        """
        Chuyển đổi một danh sách các token văn bản thành danh sách các chỉ số (index) tương ứng.
        
        Args:
            text_tokens (List[str]): Danh sách các token cần chuyển đổi.
            
        Returns:
            List[int]: Danh sách các chỉ số. Nếu token không có trong từ điển, nó sẽ được thay thế bằng index của <unk>.
        """
        return [self.stoi.get(token, self.stoi[UNK_TOKEN]) for token in text_tokens]

class TranslationDataset(Dataset):
    """
    Dataset tùy chỉnh cho bài toán dịch máy, kế thừa từ torch.utils.data.Dataset.
    Lớp này chịu trách nhiệm lưu trữ, tokenizing, và số hóa (numericalize) dữ liệu nguồn và đích.
    """
    def __init__(self, src_data: List[str], trg_data: List[str], 
                 src_tokenizer: Tokenizer, trg_tokenizer: Tokenizer,
                 src_vocab: Vocabulary, trg_vocab: Vocabulary,
                 max_strlen: int = 100):
        """
        Khởi tạo TranslationDataset. Quá trình tokenization và lọc dữ liệu được thực hiện ngay tại đây.
        
        Args:
            src_data (List[str]): Danh sách các câu văn bản nguồn.
            trg_data (List[str]): Danh sách các câu văn bản đích tương ứng.
            src_tokenizer (Tokenizer): Đối tượng Tokenizer cho ngôn ngữ nguồn.
            trg_tokenizer (Tokenizer): Đối tượng Tokenizer cho ngôn ngữ đích.
            src_vocab (Vocabulary): Từ điển cho ngôn ngữ nguồn để chuyển token thành index.
            trg_vocab (Vocabulary): Từ điển cho ngôn ngữ đích để chuyển token thành index.
            max_strlen (int): Độ dài tối đa cho phép của câu (sau khi tokenize). Các câu dài hơn sẽ bị loại bỏ. Mặc định là 100.
        """
        self.examples = []
        
        # Tokenize và Filter ngay tại lúc init để tối ưu tốc độ khi training
        print(f"Processing {len(src_data)} examples...")
        
        for s, t in zip(src_data, trg_data):
            src_tok = src_tokenizer.tokenize(s)
            trg_tok = trg_tokenizer.tokenize(t)
            
            # Lọc bỏ các cặp câu nếu một trong hai câu vượt quá độ dài tối đa
            if len(src_tok) <= max_strlen and len(trg_tok) <= max_strlen:
                # Chuyển token thành index ngay lập tức để tiết kiệm bộ nhớ RAM
                src_idx = src_vocab.numericalize(src_tok)
                trg_idx = trg_vocab.numericalize(trg_tok)
                
                # Thêm token bắt đầu <sos> và kết thúc <eos> cho câu đích
                trg_idx = [trg_vocab.stoi[SOS_TOKEN]] + trg_idx + [trg_vocab.stoi[EOS_TOKEN]]
                
                # Lưu trữ dưới dạng Tensor
                self.examples.append((torch.tensor(src_idx), torch.tensor(trg_idx)))
        
        print(f"Kept {len(self.examples)} examples after filtering.")

    def __len__(self):
        """Trả về tổng số mẫu dữ liệu trong dataset."""
        return len(self.examples)

    def __getitem__(self, idx):
        """
        Truy xuất một mẫu dữ liệu tại chỉ số index.
        
        Args:
            idx (int): Chỉ số của mẫu dữ liệu cần lấy.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Cặp tensor (source_indices, target_indices).
        """
        return self.examples[idx]

class CollateHandler:
    """
    Class xử lý việc gom nhóm (collate) các mẫu đơn lẻ thành một batch.
    Chịu trách nhiệm padding các câu trong batch về cùng độ dài.
    """
    def __init__(self, pad_idx: int):
        """
        Khởi tạo CollateHandler.
        
        Args:
            pad_idx (int): Giá trị index dùng để padding (thường là 0 cho token <pad>).
        """
        self.pad_idx = pad_idx

    def __call__(self, batch):
        """
        Hàm được DataLoader gọi để xử lý danh sách các mẫu thành batch tensor.
        
        Args:
            batch (List[Tuple[torch.Tensor, torch.Tensor]]): Danh sách các mẫu dữ liệu từ __getitem__.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - src_batch: Tensor kích thước (Batch_Size, Max_Src_Len) đã được padding.
                - trg_batch: Tensor kích thước (Batch_Size, Max_Trg_Len) đã được padding.
        """
        src_batch, trg_batch = zip(*batch)
        
        # Pad sequence để các câu trong batch có độ dài bằng nhau
        # batch_first=True giúp output có dạng (Batch, Seq_Len) thay vì (Seq_Len, Batch)
        src_batch = pad_sequence(src_batch, padding_value=self.pad_idx, batch_first=True)
        trg_batch = pad_sequence(trg_batch, padding_value=self.pad_idx, batch_first=True)
        
        return src_batch, trg_batch

def build_vocab_from_data(raw_data: List[str], tokenizer: Tokenizer, min_freq: int = 2) -> Vocabulary:
    """
    Hàm tiện ích để xây dựng Vocabulary từ danh sách các câu văn bản thô.
    
    Args:
        raw_data (List[str]): Danh sách các câu văn bản.
        tokenizer (Tokenizer): Đối tượng Tokenizer để tách từ.
        min_freq (int): Tần suất tối thiểu của từ để được đưa vào vocab. Mặc định là 2.
        
    Returns:
        Vocabulary: Đối tượng Vocabulary đã được xây dựng.
    """
    print("Building vocabulary...")
    tokens = []
    for sent in raw_data:
        tokens.extend(tokenizer.tokenize(sent))
    return Vocabulary(tokens, min_freq)

def get_data_loaders(src_file: str, trg_file: str, 
                     batch_size: int, 
                     src_vocab: Optional[Vocabulary] = None, 
                     trg_vocab: Optional[Vocabulary] = None,
                     src_lang='en', trg_lang='vi', 
                     max_strlen=100, seed=42):
    """
    Hàm chính để tạo DataLoader và Vocabulary từ file dữ liệu.
    Hỗ trợ tạo DataLoader cho cả tập Train (tự build vocab) và tập Val/Test (dùng lại vocab).
    
    Args:
        src_file (str): Đường dẫn đến file dữ liệu nguồn.
        trg_file (str): Đường dẫn đến file dữ liệu đích.
        batch_size (int): Kích thước batch.
        src_vocab (Vocabulary, optional): Vocabulary nguồn có sẵn (dùng cho Val/Test). Nếu None, sẽ tự build từ data (dùng cho Train).
        trg_vocab (Vocabulary, optional): Vocabulary đích có sẵn (dùng cho Val/Test). Nếu None, sẽ tự build từ data (dùng cho Train).
        src_lang (str): Mã ngôn ngữ nguồn ('en' hoặc 'vi'). Mặc định 'en'.
        trg_lang (str): Mã ngôn ngữ đích ('en' hoặc 'vi'). Mặc định 'vi'.
        max_strlen (int): Độ dài câu tối đa. Mặc định 100.
        seed (int): Seed ngẫu nhiên để đảm bảo tính tái lập. Mặc định 42.
        
    Returns:
        Tuple[DataLoader, Vocabulary, Vocabulary]:
            - DataLoader: Đối tượng DataLoader chứa dữ liệu đã xử lý.
            - src_vocab: Vocabulary nguồn (mới build hoặc được truyền vào).
            - trg_vocab: Vocabulary đích (mới build hoặc được truyền vào).
    """
    # 1. Thiết lập seed
    seed_everything(seed)
    
    # 2. Đọc dữ liệu thô
    src_raw, trg_raw = read_data(src_file, trg_file)
    
    # 3. Khởi tạo Tokenizers
    src_tok = Tokenizer(src_lang)
    trg_tok = Tokenizer(trg_lang)
    
    # 4. Xây dựng Vocabulary (Chỉ thực hiện nếu chưa được truyền vào - Logic tách biệt Train/Val)
    if src_vocab is None:
        src_vocab = build_vocab_from_data(src_raw, src_tok)
    if trg_vocab is None:
        trg_vocab = build_vocab_from_data(trg_raw, trg_tok)
        
    # 5. Tạo Dataset
    dataset = TranslationDataset(
        src_raw, trg_raw, 
        src_tok, trg_tok, 
        src_vocab, trg_vocab, 
        max_strlen
    )
    
    # 6. Tạo DataLoader
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=CollateHandler(pad_idx=0),
    )
    
    return dataloader, src_vocab, trg_vocab

if __name__ == "__main__":
    print("--- TRAIN PHASE ---")

    with open("test_ds.en", "w", encoding="utf-8") as f: 
        f.write("Hello world\nThis is a test")
    
    with open("test_ds.vi", "w", encoding="utf-8") as f: 
        f.write("Xin chào thế giới\nĐây là kiểm thử")

    train_loader, src_vocab, trg_vocab = get_data_loaders(
        "test_ds.en", "test_ds.vi",
        batch_size=2,
        src_vocab=None, trg_vocab=None
    )
    
    print("\n--- VAL PHASE ---")
    val_loader, _, _ = get_data_loaders(
        "test_ds.en", "test_ds.vi",
        batch_size=2,
        src_vocab=src_vocab, trg_vocab=trg_vocab
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    for src, trg in train_loader:
        src = src.to(device)
        trg = trg.to(device)
        print(f"Src on GPU: {src.is_cuda}, Shape: {src.shape}")
        print("Source Tensor:", src)
        print("Target Tensor:", trg)
        break