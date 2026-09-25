import torch

from src.shared_kernel.Config_Adapter import get_config


def ctc_greedy_decode(best: torch.Tensor, out_lengths: torch.Tensor, tokenizer) -> list[str]:
    # Takes the [B, T] argmax path, not the [B, T, V] logits: the argmax belongs on the device that
    # produced the logits, so only the ids cross to host (~1/V of the bytes).
    blank_id = get_config().model.blank_id
    texts = []
    for b in range(best.shape[0]):
        # CTC collapse: merge repeats first, THEN drop blanks, so a blank between two equal ids
        # keeps both.
        ids = torch.unique_consecutive(best[b, : int(out_lengths[b])])
        texts.append(tokenizer.decode(ids[ids != blank_id].tolist()))
    return texts
