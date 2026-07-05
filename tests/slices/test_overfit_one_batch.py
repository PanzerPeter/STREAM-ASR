import torch
import pytest

from src.slices.TrainAcousticModel.AcousticModel import AcousticModel
from src.slices.ExtractFeatures.LibriSpeechDataset import LibriSpeechDataset
from src.slices.ExtractFeatures.FeatureCollator import collate_features
from src.shared_kernel.Tokenizer_Adapter import SentencePieceTokenizer


@pytest.mark.slow
def test_overfits_two_utterances():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = SentencePieceTokenizer("data/tokenizer/bpe500.model")
    ds = LibriSpeechDataset("data/manifests/dev.jsonl", tok, train=False)
    batch = collate_features([ds[0], ds[1]])

    model = AcousticModel(cmvn_path=None).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    feats = batch.features.to(device)
    flen = batch.feature_lengths.to(device)
    toks = batch.tokens.to(device)
    tlen = batch.token_lengths.to(device)

    first = None
    for step in range(300):
        opt.zero_grad()
        logits, out_len = model(feats, flen)
        loss = model.ctc_loss(logits, out_len, toks, tlen)
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()

    assert loss.item() < 0.5 * first, f"loss did not drop: {first:.2f} -> {loss.item():.2f}"
