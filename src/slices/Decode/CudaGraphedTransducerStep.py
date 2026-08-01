# Opt-in (decode.cuda_graph) CUDA-graph capture of ONE RNN-T beam symbol step: the batched
# predictor.step + joiner.step + log_softmax. A captured graph replays as a single scheduling unit,
# collapsing the per-step kernel-launch chain (and its launch latency) into one replay + a few host
# copies. The batch is fixed at beam_size; the searcher pads its live hyps up to it and reads back
# the valid rows, so the numerics are identical to the eager path (test gates this on GPU).
import torch
import torch.nn.functional as F

from src.slices.TrainAcousticModel.TransducerModel import TransducerModel


class CudaGraphedTransducerStep:
    def __init__(self, model: TransducerModel, beam_size: int) -> None:
        self.predictor = model.predictor
        self.joiner = model.joiner
        self.device = model.ctc_head.weight.device
        self.batch = beam_size
        ctx = self.predictor.context
        enc_dim = model.encoder.output_dim
        # Persistent input buffers the graph reads from every replay (their addresses are baked into
        # the captured graph, so the caller mutates them in place rather than passing new tensors).
        self._states = torch.zeros(self.batch, ctx - 1, dtype=torch.long, device=self.device)
        self._lasts = torch.zeros(self.batch, dtype=torch.long, device=self.device)
        self._enc = torch.zeros(1, enc_dim, device=self.device)
        self._graph: torch.cuda.CUDAGraph | None = None
        self._out_logp: torch.Tensor | None = None
        self._out_state: torch.Tensor | None = None

    def _forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        pred_out, new_state = self.predictor.step(self._states, self._lasts)  # [B,D], [B,ctx-1]
        # enc stays [1, De]: the joiner adds the projected frame to every row by broadcast, exactly
        # as the eager searcher does. .contiguous() gives new_state its own graph-pool allocation so
        # the returned view can't be reclaimed between replays.
        logits = self.joiner.step(self._enc, pred_out)  # [B, V]
        return F.log_softmax(logits, dim=-1), new_state.contiguous()

    def prime(self) -> None:
        # Capture eagerly instead of on the first run(). Capture runs in CUDA's global mode, which
        # aborts if ANY other thread launches work on the device meanwhile -- so a caller that will
        # decode from several threads has to get every capture done before those threads start.
        if self._graph is None:
            self._capture()

    def _capture(self) -> None:
        # Warm up on a side stream first so lazy cuDNN/allocator init happens BEFORE capture -- any
        # allocation during capture that isn't in the graph's private pool corrupts the graph.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self._forward()
        torch.cuda.current_stream().wait_stream(side)
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._out_logp, self._out_state = self._forward()

    @torch.no_grad()
    def run(
        self, states: torch.Tensor, lasts: torch.Tensor, enc_t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # states [n, ctx-1], lasts [n], enc_t [1, De] with n <= beam_size -> (logp [n,V],
        # new_state [n, ctx-1]). First call captures lazily (params/shapes are then frozen).
        if self._graph is None:
            self._capture()
        assert self._graph is not None
        assert self._out_logp is not None and self._out_state is not None
        n = int(lasts.shape[0])
        self._states[:n].copy_(states)
        self._lasts[:n].copy_(lasts)
        self._enc.copy_(enc_t)
        self._graph.replay()
        # Clone the valid rows out of the persistent output buffers before the next replay stamps
        # them (the searcher holds child states across several steps).
        return self._out_logp[:n].clone(), self._out_state[:n].clone()
