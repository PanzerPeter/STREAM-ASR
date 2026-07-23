# src/slices/TrainLanguageModel/LmDataset.py
# Memmapped uint16 token stream -> contiguous next-token windows. nanoGPT-style: cheap random
# access, no per-item tokenization.
#
# Each window also carries per-token SEGMENT ids (line index within the window). Windows are cut at
# arbitrary offsets, so almost every one straddles several corpus lines; without segments the model
# would train on cross-line context it can never have at decode time, where a rescored hypothesis
# is always a single sentence scored from BOS. The trainer feeds these ids to the attention mask.
import numpy as np
import torch
from torch.utils.data import Dataset

from src.shared_kernel.Config_Adapter import get_config


class LmDataset(Dataset):
    def __init__(self, bin_path: str, context_len: int) -> None:
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.context_len = context_len
        self.eos = get_config().model.eos_id

    def __len__(self) -> int:
        return self.data.shape[0] - self.context_len - 1

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        window = self.data[i : i + self.context_len + 1].astype(np.int64)
        block = torch.from_numpy(window)
        x, y = block[:-1], block[1:]
        # An EOS terminates the line it belongs to, so the segment id advances on the token AFTER
        # it: cumulative EOS count minus the token's own EOS flag.
        is_eos = (x == self.eos).long()
        return x, y, is_eos.cumsum(0) - is_eos
