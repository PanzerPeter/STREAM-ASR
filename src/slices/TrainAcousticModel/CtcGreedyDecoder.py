# src/slices/TrainAcousticModel/CtcGreedyDecoder.py
import torch

from src.shared_kernel.Config_Adapter import get_config


def ctc_greedy_decode(logits: torch.Tensor, out_lengths: torch.Tensor, tokenizer) -> list[str]:
    blank_id = get_config().model.blank_id
    best = logits.argmax(dim=-1)  # [B, T]
    texts = []
    for b in range(best.shape[0]):
        prev = -1
        ids = []
        for t in range(int(out_lengths[b])):
            idx = int(best[b, t])
            if idx != prev and idx != blank_id:  # CTC collapse: drop repeats and blanks
                ids.append(idx)
            prev = idx
        texts.append(tokenizer.decode(ids))
    return texts
