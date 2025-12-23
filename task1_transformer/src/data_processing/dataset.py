import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from typing import List, Tuple
import random
import math
from .tokenizer import Tokenizer

class BidirectionalTranslationDataset(Dataset):
    """Bidirectional translation dataset with span masking"""
    
    def __init__(
        self,
        src_sentences: List[str],
        tgt_sentences: List[str],
        tokenizer: Tokenizer,
        config,
        is_training: bool = True,
    ):
        self.src_lines = src_sentences
        self.tgt_lines = tgt_sentences
        self.tokenizer = tokenizer
        self.config = config
        self.is_training = is_training
        
        # Get token IDs
        self.pad_id = tokenizer.get_token_id(config.pad_token)
        self.bos_id = tokenizer.get_token_id(config.bos_token)
        self.eos_id = tokenizer.get_token_id(config.eos_token)
        self.unk_id = tokenizer.get_token_id(config.unk_token)
        self.en_id = tokenizer.get_token_id(config.en_token)
        self.vi_id = tokenizer.get_token_id(config.vi_token)
        
        # This process creates samples with language tokens added for bidirectional training
        self.samples = []
        if is_training:
            for i, (src, tgt) in enumerate(zip(src_sentences, tgt_sentences)):
                if i % 2 == 0:
                    input_text = self.add_lang_token(src, config.vi_token)
                    target_text = tgt                                         
                    self.samples.append((input_text, target_text, "en2vi"))
                else:
                    # Output Tiếng Anh, Input Tiếng Việt
                    input_text = self.add_lang_token(tgt, config.en_token) 
                    target_text = src                                          
                    self.samples.append((input_text, target_text, "vi2en"))
        else:
            for src, tgt in zip(src_sentences, tgt_sentences):
                self.samples.append((tokenizer.add_lang_token(src, config.vi_token), tgt, "en2vi"))
        
        # Separate indices for bidirectional training
        self.en2vi_indices = [idx for idx, (_, _, d) in enumerate(self.samples) if d == "en2vi"]
        self.vi2en_indices = [idx for idx, (_, _, d) in enumerate(self.samples) if d == "vi2en"]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        src_text, tgt_text, _ = self.samples[idx]
        
        # Encode source
        src_ids = self.tokenizer.encode(src_text)[:self.config.max_len]
        
        # Encode target with BOS and EOS
        tgt_ids = [self.bos_id] + self.tokenizer.encode(tgt_text) + [self.eos_id]
        tgt_ids = tgt_ids[:self.config.max_len]
        
        # Apply span masking for training
        if self.is_training:
            src_ids = self._apply_span_masking(src_ids)
        
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)
    
    def _apply_span_masking(self, src_ids: List[int]) -> List[int]:
        """Apply span masking for better context learning"""
        if not self.is_training or random.random() > self.config.span_mask_prob:
            return src_ids
        
        src_ids = src_ids.copy()
        special = {self.pad_id, self.bos_id, self.eos_id, self.en_id, self.vi_id}
        maskable = [i for i, t in enumerate(src_ids) if t not in special]
        
        if not maskable:
            return src_ids
        
        num_to_mask = min(random.randint(1, 2), len(maskable))
        start = random.choice(maskable)
        
        for i in range(start, min(start + num_to_mask, len(src_ids))):
            if i in maskable and random.random() < 0.7:
                src_ids[i] = self.unk_id
        
        return src_ids

def collate_fn(batch):
    """Collate function for DataLoader"""
    src_list, tgt_list = zip(*batch)
    max_src = max(len(s) for s in src_list)
    max_tgt = max(len(t) for t in tgt_list)
    
    src_batch = torch.zeros(len(batch), max_src, dtype=torch.long)
    tgt_batch = torch.zeros(len(batch), max_tgt, dtype=torch.long)
    
    for i, (src, tgt) in enumerate(zip(src_list, tgt_list)):
        src_batch[i, :len(src)] = src
        tgt_batch[i, :len(tgt)] = tgt
    
    return src_batch, tgt_batch

def select_vi2en_window(indices: List[int], epoch: int, ratio: float) -> List[int]:
    """Select window of vi->en samples for this epoch"""
    if not indices or ratio <= 0:
        return []
    
    total = len(indices)
    window = max(1, int(math.ceil(total * min(ratio, 1.0))))
    start = ((epoch - 1) * window) % total
    end = start + window
    
    if end <= total:
        return indices[start:end]
    
    wrap = end - total
    return indices[start:] + indices[:wrap]

def create_data_loaders(
    train_src: List[str],
    train_tgt: List[str],
    valid_src: List[str],
    valid_tgt: List[str],
    tokenizer: Tokenizer,
    config,
    epoch: int = 1
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders"""
    
    # Create datasets
    train_dataset = BidirectionalTranslationDataset(
        train_src, train_tgt, tokenizer, config, True
    )
    valid_dataset = BidirectionalTranslationDataset(
        valid_src, valid_tgt, tokenizer, config, False
    )
    
    # Validation loader
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    
    # Training loader with bidirectional sampling
    active = list(train_dataset.en2vi_indices)
    vi_slice = select_vi2en_window(
        train_dataset.vi2en_indices, epoch, config.vi2en_epoch_ratio
    )
    active.extend(vi_slice)
    
    sampler = SubsetRandomSampler(active)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    
    return train_loader, valid_loader, len(vi_slice)