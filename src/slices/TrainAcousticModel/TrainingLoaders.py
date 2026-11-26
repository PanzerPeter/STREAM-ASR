# Train and dev loaders for the transducer stage.
import os
from typing import Any

from torch.utils.data import ConcatDataset, DataLoader

from src.shared_kernel.Config_Adapter import get_config
from src.slices.ExtractFeatures.FeatureCache import FeatureCacheReader
from src.slices.ExtractFeatures.FeatureCollator import collate_features
from src.slices.ExtractFeatures.FrameBucketSampler import FrameBucketSampler
from src.slices.ExtractFeatures.LibriSpeechDataset import LibriSpeechDataset
from src.slices.TrainAcousticModel.DevMetrics import DevSet
from src.slices.TrainAcousticModel.TransducerTrainer_Command import TransducerTrainCommand


def _open_cache(cache_dir: str, split: str) -> FeatureCacheReader | None:
    # An absent cache falls back to decoding FLAC per epoch; a present but mismatched one raises
    # in LibriSpeechDataset, because row order is the cache index.
    if not os.path.isfile(os.path.join(cache_dir, f"{split}.header.json")):
        return None
    return FeatureCacheReader(cache_dir, split)


def build_loaders(
    cmd: TransducerTrainCommand, tokenizer
) -> tuple[DataLoader, FrameBucketSampler, list[DevSet]]:
    tr = get_config().training.transducer

    def sampler_for(manifests: tuple[str, ...], shuffle: bool) -> FrameBucketSampler:
        return FrameBucketSampler(
            manifests,
            tr.max_frames_per_batch,
            shuffle=shuffle,
            seed=tr.seed,
            max_tokens_per_batch=tr.max_tokens_per_batch,
            max_lattice_per_batch=tr.max_lattice_per_batch,
        )

    # ConcatDataset and the sampler both index the sources' rows back to back in this order; each
    # dataset still checks its own cache's fingerprint.
    train_ds: ConcatDataset[Any] = ConcatDataset(
        [
            LibriSpeechDataset(src.manifest, tokenizer, _open_cache(cmd.cache_dir, src.cache_split))
            for src in cmd.train_sources
        ]
    )
    sampler = sampler_for(tuple(src.manifest for src in cmd.train_sources), shuffle=True)
    train = DataLoader(
        train_ds,
        batch_sampler=sampler,
        collate_fn=collate_features,
        num_workers=cmd.num_workers,
        pin_memory=True,
    )
    dev = [
        DevSet(
            split.name,
            DataLoader(
                LibriSpeechDataset(
                    split.manifest, tokenizer, _open_cache(cmd.cache_dir, split.cache_split)
                ),
                batch_sampler=sampler_for((split.manifest,), shuffle=False),
                collate_fn=collate_features,
            ),
        )
        for split in cmd.dev_splits
    ]
    return train, sampler, dev
