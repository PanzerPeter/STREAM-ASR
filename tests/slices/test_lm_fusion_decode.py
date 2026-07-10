import torch

from src.shared_kernel.Config_Adapter import get_config
from src.slices.Decode.CtcPrefixBeam import CtcPrefixBeam
from src.slices.Decode.AttentionRescorer import AttentionRescorer
from src.slices.Decode.LmScorer import LmScorer
from src.slices.TrainLanguageModel.StreamLmModel import StreamLmModel
from src.slices.TrainAcousticModel.AttentionDecoder import BiTransformerDecoder


def test_prefix_beam_lm_none_matches_no_scorer():
    # alpha=0 / no-LM regression lock: an lm_scorer=None beam is byte-identical to today's beam.
    torch.manual_seed(0)
    lp = torch.log_softmax(torch.randn(6, 8), dim=-1)
    a = CtcPrefixBeam(blank_id=7, beam_size=4)
    a.advance(lp)
    b = CtcPrefixBeam(blank_id=7, beam_size=4, lm_scorer=None)
    b.advance(lp)
    assert a.nbest() == b.nbest()


def test_prefix_beam_fusion_perturbs_ranking():
    # A positive LM weight must shift at least one fused prefix score away from the plain beam.
    torch.manual_seed(0)
    lp = torch.log_softmax(torch.randn(8, 10), dim=-1)  # tokens 0..8 real, blank=9
    scorer = LmScorer(StreamLmModel().eval(), weight=2.0)
    plain = CtcPrefixBeam(blank_id=9, beam_size=4)
    plain.advance(lp)
    fused = CtcPrefixBeam(blank_id=9, beam_size=4, lm_scorer=scorer)
    fused.advance(lp)
    assert dict(plain.nbest()) != dict(fused.nbest())


def test_rescorer_lm_none_matches_no_scorer():
    # alpha=0 regression lock for the second pass: lm_scorer=None reproduces the pre-LM rescore.
    torch.manual_seed(0)
    dec = BiTransformerDecoder().eval()
    memory = torch.randn(1, 12, get_config().model.encoder_dims[-1])
    mem_pad = torch.zeros(1, 12, dtype=torch.bool)
    hyps = [(5, 6, 7), (5, 6)]
    ctc = [-1.0, -1.5]
    with torch.no_grad():
        base = AttentionRescorer(dec).rescore(memory, mem_pad, hyps, ctc)
        none = AttentionRescorer(dec, lm_scorer=None).rescore(memory, mem_pad, hyps, ctc)
    assert base == none


def test_rescorer_adds_lm_sequence_score():
    # Each non-empty hypothesis' fused score = pre-LM score + the LM's weighted sequence log-prob.
    torch.manual_seed(0)
    dec = BiTransformerDecoder().eval()
    lm = LmScorer(StreamLmModel().eval(), weight=1.0)
    memory = torch.randn(1, 12, get_config().model.encoder_dims[-1])
    mem_pad = torch.zeros(1, 12, dtype=torch.bool)
    hyps = [(5, 6, 7), (5, 6)]
    ctc = [-1.0, -1.5]
    with torch.no_grad():
        base = {tuple(h): s for h, s in AttentionRescorer(dec).rescore(memory, mem_pad, hyps, ctc)}
        fused = {
            tuple(h): s for h, s in AttentionRescorer(dec, lm).rescore(memory, mem_pad, hyps, ctc)
        }
    for h in [(5, 6, 7), (5, 6)]:
        assert abs((fused[h] - base[h]) - lm.sequence_score(list(h))) < 1e-4
