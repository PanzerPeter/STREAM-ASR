import torch
from src.slices.TrainAcousticModel.AttentionDecoder import BiTransformerDecoder
from src.slices.Decode.AttentionRescorer import AttentionRescorer
from src.shared_kernel.Config_Adapter import get_config


def test_rescore_returns_sorted_scored_hyps():
    torch.manual_seed(0)
    dec = BiTransformerDecoder().eval()
    memory = torch.randn(1, 12, get_config().model.encoder_dims[-1])
    mem_pad = torch.zeros(1, 12, dtype=torch.bool)
    hyps = [(5, 6, 7), (5, 6)]
    ctc = [-1.0, -1.5]
    with torch.no_grad():
        out = AttentionRescorer(dec).rescore(memory, mem_pad, hyps, ctc)
    assert len(out) == 2
    assert [s for _, s in out] == sorted([s for _, s in out], reverse=True)
    assert isinstance(out[0][0], list)
