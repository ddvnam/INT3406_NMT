import os
import pandas as pd
from torchtext import data
import torch
from load_data import read_data
from preprocessing_data import create_fields

max_src_in_batch = 0
max_tgt_in_batch = 0

def batch_size_fn(new, count, sofar):
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)

class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches = pool(self.data(), self.random_shuffler)
        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                          self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))

def create_dataset(src_file, trg_file, max_strlen, batch_size, device, SRC, TRG, is_train=True):
    # 1. Load dữ liệu thô
    src_data, trg_data = read_data(src_file, trg_file)
    
    # Kiểm tra độ dài dữ liệu
    if len(src_data) != len(trg_data):
        print(f"Cảnh báo: Số lượng câu nguồn ({len(src_data)}) và đích ({len(trg_data)}) không bằng nhau. Sẽ lấy theo độ dài nhỏ nhất.")
        min_len = min(len(src_data), len(trg_data))
        src_data = src_data[:min_len]
        trg_data = trg_data[:min_len]

    print(f"Đang xử lý dataset (Train={is_train})... Số lượng câu: {len(src_data)}")

    # 2. Tạo DataFrame và lọc
    raw_data = {'src': src_data, 'trg': trg_data}
    df = pd.DataFrame(raw_data, columns=["src", "trg"])
    
    mask = (df['src'].str.count(' ') < max_strlen) & (df['trg'].str.count(' ') < max_strlen)
    df = df.loc[mask]
    
    temp_csv = "temp_data.csv"
    df.to_csv(temp_csv, index=False)
    
    # 3. Tạo TabularDataset
    data_fields = [('src', SRC), ('trg', TRG)]
    dataset = data.TabularDataset(
        path=temp_csv,
        format='csv',
        fields=data_fields,
        skip_header=True
    )

    if os.path.exists(temp_csv):
        os.remove(temp_csv)
    
    # 4. Build Vocabulary
    if is_train:
        print("Đang xây dựng từ điển (Vocabulary)...")
        SRC.build_vocab(dataset)
        TRG.build_vocab(dataset)
        print(f"Vocab nguồn: {len(SRC.vocab)}, Vocab đích: {len(TRG.vocab)}")

    # 5. Tạo Iterator
    iterator = MyIterator(
        dataset,
        batch_size=batch_size,
        device=device,
        repeat=False,
        sort_key=lambda x: (len(x.src), len(x.trg)),
        batch_size_fn=batch_size_fn,
        train=is_train,
        shuffle=is_train
    )
    
    return iterator