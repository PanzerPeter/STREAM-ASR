# The Decode slice's orchestration: audio -> streaming encoder -> CTC prefix-beam (live partials)
# -> attention rescore at the endpoint. Offline mode swaps the streaming encoder for one
# full-context forward but reuses both passes verbatim.
import time
from typing import Protocol

import torch
import torch.nn.functional as F

from src.shared_kernel.AudioIO_Adapter import load_audio
from src.shared_kernel.LogMel_Transform import compute_log_mel
from src.shared_kernel.Config_Adapter import get_config
from src.slices.TrainAcousticModel.HybridModel import HybridCtcAttention
from src.slices.TrainAcousticModel.StreamCache import StreamCache
from src.slices.Decode.CtcPrefixBeam import CtcPrefixBeam
from src.slices.Decode.AttentionRescorer import AttentionRescorer
from src.slices.Decode.StreamingDecode_Command import StreamingDecode_Command
from src.slices.Decode.StreamingDecode_Response import StreamingDecode_Response, SegmentResult


class _Tokenizer(Protocol):
    # Structural type: the handler only needs `.decode(ids) -> str`; it must not import a
    # concrete tokenizer implementation (SentencePiece vs. a test stub) to type-check against.
    def decode(self, ids: list[int]) -> str: ...


class StreamingDecoder_Handler:
    def __init__(self, model: HybridCtcAttention, tokenizer: _Tokenizer) -> None:
        self.model = model
        self.tok = tokenizer
        self.cfg = get_config()
        self.rescorer = AttentionRescorer(model.decoder)

    def decode(self, cmd: StreamingDecode_Command) -> StreamingDecode_Response:
        wave = load_audio(cmd.audio_path)
        # Front end is CPU-only (soundfile + torchaudio mel); move to the model's device so decode
        # runs on the GPU the CLI placed the model on. Everything downstream inherits feats.device.
        device = self.model.ctc_head.weight.device
        feats = compute_log_mel(wave).unsqueeze(0).to(device)  # [1, T, n_mels]
        audio_seconds = wave.shape[0] / self.cfg.audio.sample_rate
        start = time.perf_counter()
        if cmd.streaming:
            memory, beam, first_latency = self._stream_encode(feats, start)
        else:
            memory, beam, first_latency = self._offline_encode(feats)
        mem_pad = torch.zeros(1, memory.shape[1], dtype=torch.bool, device=memory.device)
        nbest = beam.nbest()[: self.cfg.decode.beam_size]
        rescored = self.rescorer.rescore(
            memory, mem_pad, [h for h, _ in nbest], [s for _, s in nbest]
        )
        best_ids = rescored[0][0] if rescored else []
        text = self.tok.decode(best_ids)
        seg = SegmentResult(text=text, nbest=[(self.tok.decode(h), sc) for h, sc in rescored])
        rtf = (time.perf_counter() - start) / max(audio_seconds, 1e-6)
        return StreamingDecode_Response(
            text=text, segments=[seg], rtf=rtf, first_partial_latency_s=first_latency
        )

    def _valid_out_frames(self, n_feats: int) -> int:
        # Output frames the encoder yields for n_feats real input frames: ×2 conv frontend then the
        # ×2 final downsample (ceil, matching SimpleDownsample). Used to discard the frames produced
        # by the silence padded onto the tail, so streaming stays consistent with offline.
        base = (n_feats - 1) // 2 + 1
        f = self.cfg.model.final_downsample
        return (base + f - 1) // f

    def _stream_encode(
        self, feats: torch.Tensor, start: float
    ) -> tuple[torch.Tensor, CtcPrefixBeam, float]:
        enc = self.model.encoder
        base = self.cfg.decode.chunk_size
        feat_chunk = 2 * base  # feature-rate chunk -> base base-rate frames
        n_valid = self._valid_out_frames(feats.shape[1])  # real output frames, before padding
        pad = (-feats.shape[1]) % feat_chunk
        if pad:  # trailing partial chunk: pad with edge silence so every step is aligned
            feats = F.pad(feats, (0, 0, 0, pad))
        beam = CtcPrefixBeam(self.cfg.model.blank_id, self.cfg.decode.beam_size)
        beam.reset()
        cache = StreamCache.init(enc, batch_size=1, device=feats.device)
        mems: list[torch.Tensor] = []
        first_latency = 0.0
        emitted = 0
        for i, s in enumerate(range(0, feats.shape[1], feat_chunk)):
            mem, cache = enc.streaming_forward(feats[:, s : s + feat_chunk], cache)
            take = min(mem.shape[1], n_valid - emitted)  # drop frames that come from tail padding
            if take > 0:
                mem = mem[:, :take]
                beam.advance(F.log_softmax(self.model.ctc_head(mem)[0], dim=-1))
                mems.append(mem)
                emitted += take
            if i == 0:
                first_latency = time.perf_counter() - start
        memory = (
            torch.cat(mems, dim=1)
            if mems
            else torch.zeros(1, 0, enc.output_dim, device=feats.device)
        )
        return memory, beam, first_latency

    def _offline_encode(self, feats: torch.Tensor) -> tuple[torch.Tensor, CtcPrefixBeam, float]:
        lengths = torch.tensor([feats.shape[1]], device=feats.device)
        logits, memory, out_len = self.model(feats, lengths, chunk_size=0)
        beam = CtcPrefixBeam(self.cfg.model.blank_id, self.cfg.decode.beam_size)
        beam.reset()
        beam.advance(F.log_softmax(logits[0, : int(out_len[0])], dim=-1))
        return memory, beam, 0.0
