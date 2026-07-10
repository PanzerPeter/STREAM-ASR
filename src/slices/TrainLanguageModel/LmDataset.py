# src/slices/TrainLanguageModel/LmDataset.py
# Memmapped uint16 token stream -> contiguous next-token windows. nanoGPT-style: cheap random
# access, no per-item tokenization.
import numpy as np
import torch
from torch.utils.data import Dataset


class LmDataset(Dataset):
    def __init__(self, bin_path: str, context_len: int) -> None:
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.context_len = context_len

    def __len__(self) -> int:
        return self.data.shape[0] - self.context_len - 1

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        window = self.data[i : i + self.context_len + 1].astype(np.int64)
        block = torch.from_numpy(window)
        return block[:-1], block[1:]
