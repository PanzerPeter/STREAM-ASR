import torch
from torch.utils.data import DataLoader

from src.slices.ExtractFeatures.LibriSpeechDataset import LibriSpeechDataset
from src.slices.ExtractFeatures.FeatureCollator import collate_features, IGNORE_ID
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


def test_collate_builds_bidirectional_decoder_targets():
    m = get_config().model
    mel_a = torch.randn(30, 80)
    mel_b = torch.randn(20, 80)
    batch = collate_features([(mel_a, [3, 4, 5]), (mel_b, [7, 8])])
    # dec_lengths = token_len + 1 (for the appended EOS / prepended SOS)
    assert batch.dec_lengths.tolist() == [4, 3]
    # L2R input starts with SOS; output ends with EOS at the true length, then IGNORE padding.
    assert batch.dec_in_l2r[0, 0].item() == m.sos_id
    assert batch.dec_out_l2r[0, :4].tolist() == [3, 4, 5, m.eos_id]
    assert batch.dec_out_l2r[1, 3].item() == IGNORE_ID  # padding beyond len 3 for sample b
    # R2L runs over the reversed transcript.
    assert batch.dec_out_r2l[0, :4].tolist() == [5, 4, 3, m.eos_id]
    assert batch.dec_in_r2l[0, 1:4].tolist() == [5, 4, 3]
