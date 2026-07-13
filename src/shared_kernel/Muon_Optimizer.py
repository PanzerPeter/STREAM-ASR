# Muon (Jordan): replace each 2D weight's momentum with its orthogonal polar factor (approximated by
# a fixed Newton-Schulz quintic) before the step, so every weight matrix receives a spectrally
# normalized update. Only 2D hidden weights are passed here; everything else uses AdamW. The NS
# iterate runs in fp32 for numerical stability even under bf16 autocast (SP3).
import torch


def newton_schulz_orthogonalize(g: torch.Tensor, steps: int = 5) -> torch.Tensor:
    assert g.ndim == 2, "Muon orthogonalization expects a 2D matrix"
    a, b, c = 3.4445, -4.7750, 2.0315  # quintic coefficients (Jordan)
    x = g.float()
    transpose = x.shape[0] > x.shape[1]  # iterate on the wide orientation
    if transpose:
        x = x.t()
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        aa = x @ x.t()
        x = a * x + (b * aa + c * (aa @ aa)) @ x
    if transpose:
        x = x.t()
    return x


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
        for group in self.param_groups:
            lr = group["lr"]
            mom = group["momentum"]
            ns = group["ns_steps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise ValueError("Muon only accepts 2D parameters; route others to AdamW")
                state = self.state[p]
                buf = state.get("momentum_buffer")
                if buf is None:
                    buf = state["momentum_buffer"] = torch.zeros_like(p.grad)
                buf.mul_(mom).add_(p.grad)
                update = newton_schulz_orthogonalize(buf, ns).to(p.dtype)
                # Match the update's RMS to the parameter's fan geometry (Jordan's scaling).
                scale = max(1.0, p.shape[0] / p.shape[1]) ** 0.5
                if wd:
                    p.mul_(1.0 - lr * wd)
                p.add_(update, alpha=-lr * scale)
        return loss
