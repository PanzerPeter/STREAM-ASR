import math

import numpy as np
import soundfile as sf
import torch

from src.shared_kernel.AudioIO_Adapter import load_audio
from src.shared_kernel.Config_Adapter import get_config
from src.shared_kernel.LogMel_Transform import compute_log_mel
from src.slices.Decode.StreamingDecode_Command import StreamingDecode_Command
from src.slices.Decode.StreamingDecoder_Handler import StreamingDecoder_Handler
from src.slices.Decode.TransducerBeamSearch import TransducerBeamSearch
from src.slices.TrainAcousticModel.TransducerModel import TransducerModel


class _StubTok:
    def decode(self, ids):
        return " ".join(str(i) for i in ids)


def _write_wav(tmp_path):
    sr = 16000
    wav = (np.random.randn(sr) * 0.01).astype("float32")  # 1 s
    p = tmp_path / "u.flac"
    sf.write(p, wav, sr)
    return str(p)


def _aligned_wave(feat_chunk: int) -> torch.Tensor:
    # compute_log_mel yields T = n_samples // hop_length + 1 frames (torchaudio MelSpectrogram,
    # center=True). Pick n_samples so T lands exactly on a multiple of feat_chunk (itself a
    # multiple of 2*chunk_lcm() given decode.chunk_size is a chunk_lcm() multiple), so
    # _stream_encode's tail-padding branch never triggers -- no odd-tail approximation is involved
    # and the streaming path reproduces the chunked forward frame-for-frame.
    hop = get_config().audio.hop_length
    target_frames = feat_chunk * 2  # two full streaming steps, well short of being slow
    n_samples = (target_frames - 1) * hop
    return (torch.randn(n_samples) * 0.01).float()


def test_streaming_and_offline_paths_run(tmp_path):
    torch.manual_seed(0)
    model = TransducerModel(cmvn_path=None).eval()
    handler = StreamingDecoder_Handler(model, _StubTok(), fuse_lm=False)
    path = _write_wav(tmp_path)
    with torch.no_grad():
        s = handler.decode(StreamingDecode_Command(audio_path=path, streaming=True))
        o = handler.decode(StreamingDecode_Command(audio_path=path, streaming=False))
    assert isinstance(s.text, str) and isinstance(o.text, str)
    assert s.rtf > 0 and math.isfinite(o.rtf)
    assert s.first_partial_latency_s >= 0 and o.first_partial_latency_s == 0.0
    assert len(s.segments) >= 1 and len(o.segments) >= 1


def test_lm_weight_zero_skips_lm_load():
    # lm_weight == 0 must keep lm_scorer None without ever touching the LM checkpoint on disk --
    # the alpha=0 regression lock (the pure acoustic transducer stays byte-identical).
    torch.manual_seed(0)
    model = TransducerModel(cmvn_path=None).eval()
    handler = StreamingDecoder_Handler(model, _StubTok(), lm_weight=0.0)
    assert handler.lm_scorer is None


def test_fuse_lm_gate_loads_only_when_requested(monkeypatch):
    # fuse_lm=True with lm_weight>0 must build the scorer; fuse_lm=False must not, even with the
    # same positive lm_weight. Checkpoint-free: _load_lm is stubbed to a sentinel.
    torch.manual_seed(0)
    model = TransducerModel(cmvn_path=None).eval()
    loaded: list[float] = []

    def _fake_load_lm(self):
        loaded.append(self.lm_weight)
        return object()  # sentinel scorer; never invoked during construction

    monkeypatch.setattr(StreamingDecoder_Handler, "_load_lm", _fake_load_lm)

    on = StreamingDecoder_Handler(model, _StubTok(), fuse_lm=True, lm_weight=0.3)
    assert on.lm_weight == 0.3 and on.lm_scorer is not None and loaded == [0.3]

    off = StreamingDecoder_Handler(model, _StubTok(), fuse_lm=False, lm_weight=0.3)
    assert off.lm_scorer is None and loaded == [0.3]  # gate off -> _load_lm not called again


def test_nbest_for_rescore_separates_acoustic_and_lm(tmp_path, monkeypatch):
    # Rescore-mode tuning contract: nbest_for_rescore runs the beam LM-OFF (so the hypothesis set +
    # acoustic scores are alpha-independent, byte-identical to a plain acoustic search over the same
    # memory) and attaches the *unweighted* LM sequence logprob per hypothesis for external alpha
    # rescoring. A stub scorer (raw logprob = -len(ids)) proves the wiring without a real LM.
    torch.manual_seed(0)
    model = TransducerModel(cmvn_path=None).eval()

    class _StubScorer:
        def raw_sequence_logprob(self, ids: list[int]) -> float:
            return -float(len(ids))

    monkeypatch.setattr(StreamingDecoder_Handler, "_load_lm", lambda self: _StubScorer())
    handler = StreamingDecoder_Handler(model, _StubTok(), beam_size=4, fuse_lm=True, lm_weight=1.0)
    path = _write_wav(tmp_path)
    with torch.no_grad():
        nb = handler.nbest_for_rescore(StreamingDecode_Command(audio_path=path, streaming=False))
        wave = load_audio(path)
        mem, _ = handler._encode(wave, False, 0.0)
        ref = TransducerBeamSearch(model, 4, handler.cfg.decode.max_symbols).search(mem)
    assert [(ids, ac) for ids, ac, _ in nb] == ref  # acoustic beam == pure LM-off search
    assert all(lm == -float(len(ids)) for ids, _, lm in nb)  # unweighted LM score attached


def test_nbest_for_rescore_requires_lm():
    torch.manual_seed(0)
    model = TransducerModel(cmvn_path=None).eval()
    handler = StreamingDecoder_Handler(model, _StubTok(), fuse_lm=False)  # no LM attached
    try:
        handler.nbest_for_rescore(StreamingDecode_Command(audio_path="x", streaming=False))
    except ValueError:
        return
    raise AssertionError("nbest_for_rescore must reject a handler with no LM")


def test_streaming_greedy_equals_chunked_forward_greedy():
    # Load-bearing equivalence: the STREAMING decode path reproduces greedy decode over the
    # batched CHUNKED-forward encoder memory exactly. This is test_streaming_forward_equivalence
    # (streaming_forward(chunk) == forward(chunk_size=B)) lifted to the decode layer: with
    # beam_size=1, search() is a per-frame greedy decision (blank-continue vs. the single best
    # non-blank token), a pure function of encoder memory, so identical memory => identical text.
    #
    # We deliberately do NOT compare against the OFFLINE path here. _offline_encode uses
    # chunk_size=0 (full bidirectional context) by design -- that is the spec's lower-WER offline
    # mode and it can see future frames the causal streaming path never can, so it is intentionally
    # non-equivalent to streaming. The meaningful invariant is streaming == the SAME chunk-causal
    # computation done batched, which is what CHUNK=decode.chunk_size gives below.
    torch.manual_seed(0)
    model = TransducerModel(cmvn_path=None).eval()
    handler = StreamingDecoder_Handler(model, _StubTok(), beam_size=1, fuse_lm=False)
    chunk = handler.cfg.decode.chunk_size  # base-rate chunk the streaming path feeds
    feat_chunk = 2 * chunk
    wave = _aligned_wave(feat_chunk)
    feats = compute_log_mel(wave).unsqueeze(0)
    lengths = torch.tensor([feats.shape[1]])
    with torch.no_grad():
        s = handler.decode_waveform(wave, streaming=True)
        # Batched chunk-causal reference memory (same mask streaming induces), decoded via the
        # same greedy searcher the streaming path used.
        mem_chunked, out_len = model.encoder(feats, lengths, chunk_size=chunk)
        ref_ids = handler.searcher.search(mem_chunked[:, : int(out_len[0])])[0][0]
    assert s.text == handler.tok.decode(ref_ids)
