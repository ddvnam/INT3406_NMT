# src/evaluation/metrics.py
import torch
from tqdm import tqdm
from sacrebleu.metrics import BLEU, CHRF
from typing import Dict, List, Tuple
from .decoding import greedy_decode

def evaluate(
    model,
    criterion,
    tokenizer,
    valid_src: List[str],
    valid_tgt: List[str],
    config,
    max_samples: int = 500
) -> Dict[str, float]:
    """Evaluate model on validation set"""
    model.eval()
    
    # Initialize metrics
    metric_bleu = BLEU(effective_order=True)
    metric_chrf = CHRF()
    
    # Prepare data
    from ..data_processing.dataset import BidirectionalTranslationDataset
    from ..data_processing.dataset import collate_fn
    
    valid_dataset = BidirectionalTranslationDataset(
        valid_src, valid_tgt, tokenizer, config, False
    )
    
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
    )
    
    # Evaluation
    total_loss = 0.0
    all_predictions = []
    all_references = []
    sample_count = 0
    
    pbar = tqdm(valid_loader, desc="Evaluating", leave=False)
    
    for src_batch, tgt_batch in pbar:
        src_batch = src_batch.to(config.device)
        tgt_batch = tgt_batch.to(config.device)
        
        # Calculate loss
        with torch.no_grad():
            logits = model(src_batch, tgt_batch)
            targets = tgt_batch[:, 1:]
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )
        
        total_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})
        
        # Decode for metrics
        if sample_count < max_samples:
            predictions = greedy_decode(
                model, src_batch, tokenizer, config
            )
            
            # Prepare references
            eos = tokenizer.get_token_id(config.eos_token)
            batch_refs = []
            
            for tgt_seq in tgt_batch:
                ref = tgt_seq[1:].cpu().tolist()
                if eos in ref:
                    ref = ref[:ref.index(eos)]
                batch_refs.append(tokenizer.decode(ref))
            
            all_predictions.extend(predictions)
            all_references.extend(batch_refs)
            sample_count += len(predictions)
    
    # Calculate metrics
    avg_loss = total_loss / len(valid_loader)
    
    metrics = {"loss": avg_loss, "bleu": 0.0, "chrf": 0.0}
    
    if all_predictions:
        try:
            score_bleu = metric_bleu.corpus_score(all_predictions, [all_references])
            score_chrf = metric_chrf.corpus_score(all_predictions, [all_references])
            
            metrics["bleu"] = score_bleu.score
            metrics["chrf"] = score_chrf.score
        except Exception as e:
            print(f"Metric calculation failed: {e}")
    
    return metrics