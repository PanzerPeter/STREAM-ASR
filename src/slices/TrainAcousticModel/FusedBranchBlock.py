import torch
import torch.nn as nn
import torch.nn.functional as F

from src.shared_kernel.Config_Adapter import get_config
from src.shared_kernel.RoPE_Transform import apply_rotary, rope_inv_freq, rotary_tables
from src.slices.TrainAcousticModel.StreamCache import BlockCache

_EPS = 1e-6


def mlp_width(dim: int) -> int:
    # 8/3 d rounded DOWN to a multiple of 8: the conv-gated MLP's hidden width (680/1024/1360).
    return int(8 * dim / 3) // 8 * 8


def rms_norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.shape[-1],), eps=_EPS)


def _uniform(rows: int, cols: int, fan_in: int) -> nn.Parameter:
    # nn.Linear's default init, applied per slice against the fused GEMM's true fan-in.
    bound = fan_in**-0.5
    return nn.Parameter(torch.empty(rows, cols).uniform_(-bound, bound))


class FusedBranchBlock(nn.Module):
    """Gated attention, then a conv-gated MLP on its result: two residual steps, each a Peri-LN
    update with its own floored gate (MODEL_ARCHITECTURE.md §4.2).

        q,k,v,g = rms_norm(x) @ W_in[q,k,v,g]
        a = SDPA(rope(qk_norm(q)), rope(qk_norm(k)), v) * sigmoid(g)
        x = x + beta_attn * rms_norm(a @ W_out[a])
        u,z = rms_norm(x) @ W_in[u,z];  c = dwconv_causal(u) * silu(z)
        x = x + beta * rms_norm(c @ W_out[c]),  beta = floor + (1 - floor) * sigmoid(b)

    W_in and W_out are held as one parameter per logical slice and concatenated in forward, so Muon
    orthogonalises q, k, v, g, u, z (and W_out's attention / MLP slices) each on its own scale.
    Every config value is read here, never in forward: the block is the unit torch.compile sees.
    """

    inv_freq: torch.Tensor

    def __init__(self, dim: int) -> None:
        super().__init__()
        m = get_config().model
        self.dim = dim
        self.hf = mlp_width(dim)
        self.head_dim = m.head_dim
        self.heads = dim // m.head_dim
        self.kernel = m.conv_kernel_size
        self.floor = m.gate_floor
        # Creation order is the init RNG order: kept as the release run drew it.
        self.w_in = nn.ParameterDict(
            {n: _uniform(self.hf if n in ("u", "z") else dim, dim, dim) for n in "qkvguz"}
        )
        fan_out_in = self.hf + dim
        self.w_out = nn.ParameterDict(
            {n: _uniform(dim, dim if n == "a" else self.hf, fan_out_in) for n in "ac"}
        )
        self.conv = nn.Conv1d(self.hf, self.hf, self.kernel, groups=self.hf)
        self.q_gain = nn.Parameter(torch.ones(self.head_dim))
        self.k_gain = nn.Parameter(torch.ones(self.head_dim))
        self.register_buffer(
            "inv_freq", rope_inv_freq(self.head_dim, m.rope_base), persistent=False
        )
        self.gate = nn.Parameter(torch.full((dim,), m.gate_init))
        self.gate_attn = nn.Parameter(torch.full((dim,), m.gate_init))

    def beta(self, attn: bool = False) -> torch.Tensor:
        gate = self.gate_attn if attn else self.gate
        return self.floor + (1.0 - self.floor) * torch.sigmoid(gate)

    def _heads(self, x: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        x = x.view(b, t, self.heads, self.head_dim)
        # The gain in the activation dtype keeps the fused rms_norm kernel under bf16 autocast.
        return F.rms_norm(x, (self.head_dim,), gain.to(x.dtype), _EPS).transpose(1, 2)

    def _project(self, x: torch.Tensor, names: str) -> list[torch.Tensor]:
        w = torch.cat([self.w_in[n] for n in names])
        sizes = [self.hf if n in ("u", "z") else self.dim for n in names]
        return list(F.linear(rms_norm(x), w).split(sizes, dim=-1))

    def _add(self, x: torch.Tensor, y: torch.Tensor, attn: bool = False) -> torch.Tensor:
        return x + self.beta(attn) * rms_norm(y)

    def _finish(
        self, x: torch.Tensor, a: torch.Tensor, hist: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The attention step's residual update, then the MLP step. Returns the block output and
        the conv's next left context; hist is the kernel-1 frames of u before this call's (None =
        start of sequence, zeros)."""
        x = self._add(x, F.linear(a, self.w_out["a"]), attn=True)
        u, z = self._project(x, "uz")
        pad = self.kernel - 1
        u_hist = F.pad(u, (0, 0, pad, 0)) if hist is None else torch.cat([hist, u], dim=1)
        c = self.conv(u_hist.transpose(1, 2)).transpose(1, 2) * F.silu(z)
        return self._add(x, F.linear(c, self.w_out["c"])), u_hist[:, u_hist.shape[1] - pad :]

    def _gated_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        q_pos: torch.Tensor,
        k_pos: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        cos_q, sin_q = rotary_tables(q_pos, self.inv_freq, q.dtype)
        cos_k, sin_k = rotary_tables(k_pos, self.inv_freq, k.dtype)
        q, k = apply_rotary(q, cos_q, sin_q), apply_rotary(k, cos_k, sin_k)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        b, _, t, _ = out.shape
        return out.transpose(1, 2).reshape(b, t, self.dim) * torch.sigmoid(g)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """x [B, T, d]; mask [B, 1, T or 1, T] bool, True = attend."""
        q, k, v, g = self._project(x, "qkvg")
        pos = torch.arange(x.shape[1], device=x.device)
        v_h = v.view(v.shape[0], v.shape[1], self.heads, self.head_dim).transpose(1, 2)
        a = self._gated_attention(
            self._heads(q, self.q_gain), self._heads(k, self.k_gain), v_h, g, pos, pos, mask
        )
        return self._finish(x, a, None)[0]

    def streaming_forward(
        self, x: torch.Tensor, cache: BlockCache, valid: torch.Tensor, left: int
    ) -> tuple[torch.Tensor, BlockCache]:
        """One chunk. valid [B, C] marks real frames; left is this rate's history (-1 unlimited)."""
        q, k, v, g = self._project(x, "qkvg")
        k_all = torch.cat([cache.k, self._heads(k, self.k_gain)], dim=2)
        v_h = v.view(v.shape[0], v.shape[1], self.heads, self.head_dim).transpose(1, 2)
        v_all = torch.cat([cache.v, v_h], dim=2)
        valid_all = torch.cat([cache.key_valid, valid], dim=1)
        n_keys, n_q = k_all.shape[2], x.shape[1]
        k_pos = torch.arange(n_keys, device=x.device)
        mask = attention_mask(valid_all, valid, None).expand(-1, -1, n_q, -1).clone()
        # A padded query attends to itself as well, exactly as batched forward lets it.
        mask[:, 0, torch.arange(n_q), n_keys - n_q + torch.arange(n_q)] |= ~valid
        a = self._gated_attention(
            self._heads(q, self.q_gain), k_all, v_all, g, k_pos[n_keys - n_q :], k_pos, mask
        )
        keep = n_keys if left < 0 else min(n_keys, left)
        out, conv = self._finish(x, a, cache.conv)
        new = BlockCache(
            conv=conv.detach(),
            k=k_all[:, :, n_keys - keep :].detach(),
            v=v_all[:, :, n_keys - keep :].detach(),
            key_valid=valid_all[:, n_keys - keep :],
        )
        return out, new


def attention_mask(
    key_valid: torch.Tensor, query_valid: torch.Tensor, window: torch.Tensor | None
) -> torch.Tensor:
    """[B, 1, Tq or 1, Tk] bool, True = attend: real keys inside the chunk window.

    A padded query whose window holds no real key would see an all-False row, and SDPA returns NaN
    for it -- NaN that a zero attention weight then carries into every REAL query that reads the
    padded frame's value (0 * NaN). So a padded query may always attend to itself; real queries
    always see at least themselves already.
    """
    mask = key_valid[:, None, None, :]
    if window is None:
        return mask
    mask = mask & window[None, None]
    eye = torch.eye(window.shape[0], dtype=torch.bool, device=window.device)
    return mask | (eye[None, None] & ~query_valid[:, None, :, None])
