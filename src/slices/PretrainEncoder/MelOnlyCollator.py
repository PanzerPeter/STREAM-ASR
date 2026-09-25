# Zero-pads a batch of ragged [T, 80] log-mel tensors to [B, Tmax, 80]. BEST-RQ masks/quantizes
# on the padded grid, so lengths travel alongside to exclude pad frames from the pretrain loss.
import torch
from torch.nn.utils.rnn import pad_sequence


def collate_mels(samples: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([m.shape[0] for m in samples], dtype=torch.long)
    return pad_sequence(samples, batch_first=True).float(), lengths
