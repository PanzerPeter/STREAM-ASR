import torch
from torch.utils.data import DataLoader

from src.slices.ExtractFeatures.LibriSpeechDataset import LibriSpeechDataset
from src.slices.ExtractFeatures.FeatureCollator import collate_features
from src.slices.ExtractFeatures.FeatureBatch_Response import FeatureBatch
from src.shared_kernel.Tokenizer_Adapter import SentencePieceTokenizer
from src.shared_kernel.Config_Adapter import get_config


def test_collate_produces_padded_batch():
    tok = SentencePieceTokenizer("data/tokenizer/bpe500.model")
    ds = LibriSpeechDataset("data/manifests/dev.jsonl", tok, train=False)
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_features)

    batch = next(iter(loader))
    assert isinstance(batch, FeatureBatch)
    assert batch.features.ndim == 3 and batch.features.shape[2] == get_config().audio.n_mels
    assert batch.features.shape[0] == 4
    assert batch.feature_lengths.max().item() == batch.features.shape[1]
    assert batch.tokens.shape[0] == 4
    assert batch.token_lengths.dtype == torch.long
