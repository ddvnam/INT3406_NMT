# src/evaluation/decoding.py
import torch
import torch.nn.functional as F
from typing import List

@torch.no_grad()
def greedy_decode(
    model,
    src_ids: torch.Tensor,
    tokenizer,
    config,
    max_len: int = 128
) -> List[str]:
    """Greedy decoding for validation"""
    model.eval()
    batch_size = src_ids.size(0)
    device = src_ids.device
    
    # Token IDs
    bos = tokenizer.get_token_id(config.bos_token)
    eos = tokenizer.get_token_id(config.eos_token)
    pad = tokenizer.get_token_id(config.pad_token)
    
    # Encode source
    src_pad_mask = (src_ids == pad)
    src_emb = model.embedding(src_ids) * model.emb_scale
    src_input = model.emb_dropout(src_emb)
    
    enc_out = src_input
    for layer in model.encoder_layers:
        enc_out = layer(enc_out, src_pad_mask)
    enc_out = model.encoder_final_norm(enc_out)
    
    # Initialize decoder
    tgt_ids = torch.full((batch_size, 1), bos, dtype=torch.long, device=device)
    
    # Decode step by step
    for _ in range(max_len):
        tgt_pad = (tgt_ids == pad)
        tgt_len = tgt_ids.size(1)
        
        # Causal mask
        tgt_causal = torch.triu(
            torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=device),
            diagonal=1
        )
        
        # Decoder forward
        tgt_emb = model.embedding(tgt_ids) * model.emb_scale
        tgt_input = model.emb_dropout(tgt_emb)
        
        dec_out = tgt_input
        for layer in model.decoder_layers:
            dec_out = layer(dec_out, enc_out, tgt_pad, tgt_causal, src_pad_mask)
        dec_out = model.decoder_final_norm(dec_out)
        
        # Get next token
        logits = F.linear(dec_out, model.embedding.weight, model.output_bias)
        next_tokens = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        tgt_ids = torch.cat([tgt_ids, next_tokens], dim=1)
        
        # Stop if all sequences have EOS
        if (next_tokens.squeeze(-1) == eos).all():
            break
    
    # Decode to text
    decoded = []
    for seq in tgt_ids:
        tokens = seq[1:].cpu().tolist()
        if eos in tokens:
            tokens = tokens[:tokens.index(eos)]
        decoded.append(tokenizer.decode(tokens))
    
    return decoded

@torch.no_grad()
def beam_search_decode(
    model,
    src_ids: torch.Tensor,
    tokenizer,
    config,
    beam_size: int = 3,
    max_len: int = 128,
    length_penalty: float = 0.6
) -> List[str]:
    """Beam search decoding"""
    model.eval()
    device = src_ids.device
    batch_size = src_ids.size(0)
    
    # Token IDs
    bos = tokenizer.get_token_id(config.bos_token)
    eos = tokenizer.get_token_id(config.eos_token)
    pad = tokenizer.get_token_id(config.pad_token)
    
    # Encode source (once)
    src_pad_mask = (src_ids == pad)
    src_emb = model.embedding(src_ids) * model.emb_scale
    src_input = model.emb_dropout(src_emb)
    
    enc_out = src_input
    for layer in model.encoder_layers:
        enc_out = layer(enc_out, src_pad_mask)
    enc_out = model.encoder_final_norm(enc_out)
    
    decoded_sentences = []
    
    # Beam search for each sample
    for b in range(batch_size):
        single_enc = enc_out[b:b+1]
        single_mask = src_pad_mask[b:b+1]
        
        # Initialize beams
        beams = [(0.0, [bos])]
        completed_beams = []
        
        for _ in range(max_len):
            new_beams = []
            
            for score, tokens in beams:
                # Skip completed beams
                if tokens[-1] == eos:
                    completed_beams.append((score, tokens))
                    continue
                
                # Decoder forward
                tgt_ids = torch.tensor([tokens], dtype=torch.long, device=device)
                tgt_pad = (tgt_ids == pad)
                tgt_len = tgt_ids.size(1)
                
                tgt_causal = torch.triu(
                    torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=device),
                    diagonal=1
                )
                
                tgt_emb = model.embedding(tgt_ids) * model.emb_scale
                tgt_input = model.emb_dropout(tgt_emb)
                
                dec_out = tgt_input
                for layer in model.decoder_layers:
                    dec_out = layer(dec_out, single_enc, tgt_pad, tgt_causal, single_mask)
                dec_out = model.decoder_final_norm(dec_out)
                
                # Get probabilities
                logits = F.linear(dec_out[:, -1, :], model.embedding.weight, model.output_bias)
                log_probs = F.log_softmax(logits, dim=-1)
                
                # Top candidates
                top_scores, top_indices = torch.topk(log_probs, beam_size * 2)
                
                for v, i in zip(top_scores[0], top_indices[0]):
                    new_score = score + v.item()
                    new_tokens = tokens + [i.item()]
                    new_beams.append((new_score, new_tokens))
            
            # Prune beams
            if not new_beams:
                break
            
            beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_size]
        
        # Get best beam with length penalty
        if beams:
            completed_beams.extend(beams)
        
        best_beam = max(
            completed_beams,
            key=lambda x: x[0] / (len(x[1]) ** length_penalty)
        )
        
        # Decode tokens
        final_tokens = best_beam[1]
        if final_tokens[0] == bos:
            final_tokens = final_tokens[1:]
        if eos in final_tokens:
            final_tokens = final_tokens[:final_tokens.index(eos)]
        
        decoded_sentences.append(tokenizer.decode(final_tokens))
    
    return decoded_sentences