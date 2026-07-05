# src/slices/ExtractFeatures/LibriSpeechDataset.py
import json
import random

from torch.utils.data import Dataset

from src.shared_kernel.AudioIO_Adapter import load_audio
from src.shared_kernel.Config_Adapter import get_config
from src.shared_kernel.LogMel_Transform import compute_log_mel
from src.slices.ExtractFeatures.SpecAugment_Transform import apply_spec_augment
from src.slices.ExtractFeatures.SpeedPerturb_Transform import apply_speed_perturb


class LibriSpeechDataset(Dataset):
    def __init__(self, manifest: str, tokenizer, train: bool) -> None:
        self._rows = [json.loads(line) for line in open(manifest, encoding="utf-8")]
        self._tokenizer = tokenizer
        self._train = train

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int):
        row = self._rows[index]
        wave = load_audio(row["audio_filepath"])

        if self._train:
            factors = get_config().augment.speed_perturb_factors
            wave = apply_speed_perturb(wave, random.choice(factors))

        mel = compute_log_mel(wave)

        if self._train:
            mel = apply_spec_augment(mel)

        token_ids = self._tokenizer.encode(row["text"])
        return mel, token_ids
