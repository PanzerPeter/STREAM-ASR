# Muon (Jordan): replace each 2D weight's momentum with its orthogonal polar factor (approximated by
# a fixed Newton-Schulz quintic) before the step, so every weight matrix receives a spectrally
# normalized update. Only 2D hidden weights are passed here; everything else uses AdamW. The NS
# iterate runs in fp32 for numerical stability even under bf16 autocast.
#
# The iteration is batched over same-shaped matrices: it depends on neither the learning rate nor
# weight decay, so every matrix of one shape orthogonalizes in a single bmm chain. The transducer's
# 135 Muon matrices fall into 22 shape classes, turning ~2700 small kernel launches per optimizer
# step into ~440 -- the same arithmetic, issued in bulk. (Running the iteration in bf16 would be
# faster still and is what Jordan's reference does, but measured on this model it moves the update
# direction by ~7% relative to fp64 where fp32/TF32 moves it by 0.09%, so it stays in fp32.)
import torch

# Jordan's quintic coefficients.
_A, _B, _C = 3.4445, -4.7750, 2.0315


def _newton_schulz_batched(g: torch.Tensor, steps: int) -> torch.Tensor:
    # g: [N, rows, cols] -- N matrices of one shape. Returns the same stack, spectrally normalized.
    x = g.float()
    transpose = x.shape[-2] > x.shape[-1]  # iterate on the wide orientation
    if transpose:
        x = x.transpose(-2, -1)
    x = x / (torch.linalg.matrix_norm(x).view(-1, 1, 1) + 1e-7)
    for _ in range(steps):
        aa = x @ x.transpose(-2, -1)
        x = _A * x + (_B * aa + _C * (aa @ aa)) @ x
    return x.transpose(-2, -1) if transpose else x


def newton_schulz_orthogonalize(g: torch.Tensor, steps: int = 5) -> torch.Tensor:
    assert g.ndim == 2, "Muon orthogonalization expects a 2D matrix"
    return _newton_schulz_batched(g.unsqueeze(0), steps).squeeze(0)


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ) -> None:
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        # Bucket by (shape, ns_steps) -- everything the Newton-Schulz iteration depends on. The
        # per-parameter lr/weight-decay ride along and are applied after, grouped so the writes go
        # through the multi-tensor kernels too.
        buckets: dict[tuple[tuple[int, ...], int], list[tuple[torch.nn.Parameter, float, float]]]
        buckets = {}
        momentum_groups: dict[float, tuple[list[torch.Tensor], list[torch.Tensor]]] = {}
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise ValueError("Muon only accepts 2D parameters; route others to AdamW")
                state = self.state[p]
                buf = state.get("momentum_buffer")
                if buf is None:
                    buf = state["momentum_buffer"] = torch.zeros_like(p.grad)
                bufs, grads = momentum_groups.setdefault(group["momentum"], ([], []))
                bufs.append(buf)
                grads.append(p.grad)
                key = (tuple(p.shape), group["ns_steps"])
                buckets.setdefault(key, []).append((p, group["lr"], group["weight_decay"]))

        for momentum, (bufs, grads) in momentum_groups.items():
            torch._foreach_mul_(bufs, momentum)
            torch._foreach_add_(bufs, grads)

        for (shape, ns_steps), items in buckets.items():
            stacked = torch.stack([self.state[p]["momentum_buffer"] for p, _, _ in items])
            updates = _newton_schulz_batched(stacked, ns_steps)
            # Match the update's RMS to the parameter's fan geometry (Jordan's scaling). Shape is
            # constant within a bucket, so this folds into the batched multiply.
            updates = updates.mul_(max(1.0, shape[0] / shape[1]) ** 0.5)
            applies: dict[tuple[float, float], tuple[list, list]] = {}
            for (p, lr, weight_decay), update in zip(items, updates):
                params, deltas = applies.setdefault((lr, weight_decay), ([], []))
                params.append(p)
                deltas.append(update.to(p.dtype))
            for (lr, weight_decay), (params, deltas) in applies.items():
                if weight_decay:
                    torch._foreach_mul_(params, 1.0 - lr * weight_decay)
                torch._foreach_add_(params, deltas, alpha=-lr)
        return loss
