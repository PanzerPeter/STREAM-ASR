# Memmapped uint16 token stream -> non-overlapping next-token windows. Window i covers tokens
# [i * context_len, (i + 1) * context_len], so one pass over the dataset reads every token once:
# the trainer shuffles window indices without replacement, and validation scores the whole bin.
#
# Each window also carries per-token SEGMENT ids (line index within the window) for the document
# mask. The EOS that ends a line opens the next one: segment ids advance ON each EOS, so every
# packed line trains as [EOS] + line, the exact row the rescorer scores a hypothesis as. (Grouping
# the EOS with the line it ends instead trains the first token of every line on the previous line's
# context and never shows any token the EOS key it attends at decode.)
import numpy as np
import torch
from torch.utils.data import Dataset

from src.shared_kernel.Config_Adapter import get_config


class LmDataset(Dataset):
    def __init__(self, bin_path: str, context_len: int) -> None:
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.context_len = context_len
        self.eos = get_config().lm.eos_id
        self.windows = (self.data.shape[0] - 1) // context_len
        if self.windows < 1:
            raise ValueError(
                f"{bin_path}: {self.data.shape[0]} tokens, needs more than context_len "
                f"{context_len} for one window"
            )

    def __len__(self) -> int:
        return self.windows

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start = i * self.context_len
        window = self.data[start : start + self.context_len + 1].astype(np.int64)
        block = torch.from_numpy(window)
        x, y = block[:-1], block[1:]
        return x, y, (x == self.eos).long().cumsum(0)
