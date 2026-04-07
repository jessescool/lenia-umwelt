# substrate/lenia.py — Jesse Cool (jessescool)
"""Core Lenia environment primitives and kernel utilities."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.fft import rfft2, irfft2

torch.backends.cudnn.benchmark = True


def _auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

ArrayLike = torch.Tensor | Sequence[Sequence[float]]


@dataclass
class Config:
    grid_shape: Tuple[int, int] = (48, 48)
    dt: float = 0.05
    mu: float = 0.15
    sigma: float = 0.015
    kernel_radius: int = 5
    growth_type: int = 2
    kernel: torch.Tensor | None = None
    device: torch.device = _auto_device()
    dtype: torch.dtype = torch.float32
    timescale_T: float = 10.0  # Creature timescale (dt = 1/T)
    kn: int = 1  # Kernel core function type
    kernel_params: dict | None = None  # Raw params for kernel rebuild on rescale
    scale: int = 1  # Scale factor applied (1 = base resolution)
    base_grid: int | None = None  # Original grid size before scaling

    @classmethod
    def from_animal(
        cls,
        animal,
        base_grid: int | Tuple[int, int] | None = None,
        scale: int = 1,
        *,
        grid_shape: int | Tuple[int, int] | None = None,  # Backwards compat alias
    ) -> "Config":
        """Build a Config from an Animal's params and desired grid size."""
        # grid_shape= alias for base_grid
        if base_grid is None and grid_shape is not None:
            base_grid = grid_shape
        elif base_grid is None:
            raise TypeError("from_animal() missing required argument: 'base_grid'")

        def _int(value, default):
            try:
                return int(float(value))
            except (TypeError, ValueError):
                return default
        def _float(value, default):
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        params = animal.params or {}
        base_radius = _int(params.get("R", 13), 13)
        timestep = _float(params.get("T", 10.0), 10.0)
        mu = _float(params.get("m", 0.15), 0.15)
        sigma = _float(params.get("s", 0.015), 0.015)
        growth_type = _int(params.get("gn", 2), 2)
        kn = _int(params.get("kn", 1), 1)

        # Support both int (square) and tuple (rectangular) grids
        if isinstance(base_grid, int):
            base_h, base_w = base_grid, base_grid
        else:
            base_h, base_w = base_grid[0], base_grid[1]

        final_h, final_w = base_h * scale, base_w * scale
        scaled_radius = base_radius * scale

        kernel_size = 2 * scaled_radius + 1
        kernel = build_kernel(
            (kernel_size, kernel_size),
            radius=scaled_radius,
            kn=kn,
            params=params,
        )

        return cls(
            grid_shape=(final_h, final_w),
            dt=1.0 / max(timestep, 1.0),
            mu=mu,
            sigma=sigma,
            kernel_radius=scaled_radius,
            growth_type=growth_type,
            kernel=kernel,
            timescale_T=timestep,
            kn=kn,
            kernel_params=params,
            scale=scale,
            base_grid=base_h,  # Store height as base_grid for backwards compat
        )


class Board:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.shape = cfg.grid_shape
        self._tensor = torch.zeros(self.shape, dtype=cfg.dtype, device=cfg.device)

    @property
    def tensor(self) -> torch.Tensor:
        return self._tensor

    @property
    def cells(self) -> torch.Tensor:
        return self._tensor

    @cells.setter
    def cells(self, value: ArrayLike | torch.Tensor) -> None:
        data = torch.as_tensor(value, dtype=self.cfg.dtype, device=self.cfg.device)
        if tuple(data.shape) != tuple(self.shape):
            raise ValueError("Incoming board data has mismatched shape")
        self._tensor.copy_(data)

    def replace_tensor(self, new: torch.Tensor) -> None:
        """Replace the backing tensor reference (must match shape)."""
        if new.shape != self._tensor.shape:
            raise ValueError(f"Shape mismatch: expected {self._tensor.shape}, got {new.shape}")
        self._tensor = new

    def to_numpy(self) -> "np.ndarray":  # type: ignore[name-defined]
        import numpy as np  # Local import to keep core torch-only

        return self._tensor.detach().cpu().numpy()

    def clear(self) -> None:
        self._tensor.zero_()

    def place(
        self,
        pattern: ArrayLike,
        *,
        position: Tuple[int, int] | None = None,
        wrap: bool = True,
    ) -> None:
        block = torch.as_tensor(pattern, dtype=self.cfg.dtype, device=self.cfg.device)
        ph, pw = block.shape
        rows, cols = self.shape
        if position is None:
            r0 = (rows - ph) // 2
            c0 = (cols - pw) // 2
        else:
            r0, c0 = position
        if wrap:
            row_idx = (torch.arange(ph, device=self.cfg.device) + r0) % rows
            col_idx = (torch.arange(pw, device=self.cfg.device) + c0) % cols
            self._tensor.index_put_((row_idx[:, None], col_idx[None, :]), block)
        else:
            r_start = max(r0, 0)
            c_start = max(c0, 0)
            r_end = min(r0 + ph, rows)
            c_end = min(c0 + pw, cols)
            if r_start >= r_end or c_start >= c_end:
                return
            src_r0 = max(0, -r0)
            src_c0 = max(0, -c0)
            src = block[src_r0:src_r0 + (r_end - r_start), src_c0:src_c0 + (c_end - c_start)]
            self._tensor[r_start:r_end, c_start:c_end] = src

class Automaton:
    def __init__(self, cfg: Config, *, fft: bool = True):
        self.cfg = cfg
        self.use_fft = fft
        if cfg.kernel is not None:
            base_kernel = cfg.kernel
        else:
            base_kernel = _make_kernel(cfg.kernel_radius, dtype=cfg.dtype, device=cfg.device)
        weight = _kernel_to_conv_weight(base_kernel, cfg)
        self.kernel_radius = weight.shape[-1] // 2
        self.kernel = weight  # keep [1,1,K,K] for inspection / backward compat

        # FFT kernel precompute (cheap, always available)
        # Embed [K, K] kernel into [H, W] grid with center at (0, 0) so that
        # rfft2 multiplication gives circular convolution (no padding needed).
        H, W = cfg.grid_shape
        K = weight.shape[-1]
        if K > min(H, W):
            raise ValueError(f"Kernel size {K} exceeds grid {H}x{W}")
        kernel_2d = weight[0, 0]                     # [K, K]
        padded_k = torch.zeros(H, W, dtype=cfg.dtype, device=cfg.device)
        padded_k[:K, :K] = kernel_2d
        # Roll so kernel center lands at (0, 0).
        # K = 2R+1 is always odd, so center = K//2 is exact (no half-pixel shift).
        center = K // 2
        padded_k = torch.roll(padded_k, shifts=(-center, -center), dims=(0, 1))
        self._kernel_fft = rfft2(padded_k)           # [H, W//2+1] complex
        self._grid_shape = (H, W)

        # Cache growth scalars as tensors to avoid re-creation every step
        self._mu_t = torch.tensor(cfg.mu, dtype=cfg.dtype, device=cfg.device)
        self._sigma_safe = max(cfg.sigma, 1e-6)

    def _rebuild_kernel_fft(self, H: int, W: int) -> torch.Tensor:
        """Rebuild FFT kernel for a different grid size (lazy, cached)."""
        if (H, W) == self._grid_shape:
            return self._kernel_fft
        K = self.kernel.shape[-1]
        kernel_2d = self.kernel[0, 0]
        padded_k = torch.zeros(H, W, dtype=self.cfg.dtype, device=self.cfg.device)
        padded_k[:K, :K] = kernel_2d
        center = K // 2
        padded_k = torch.roll(padded_k, shifts=(-center, -center), dims=(0, 1))
        kfft = rfft2(padded_k)
        # Cache for next call with same shape
        self._kernel_fft = kfft
        self._grid_shape = (H, W)
        return kfft

    def _excitation_spatial(self, state_2d: torch.Tensor) -> torch.Tensor:
        """Excitation via spatial F.conv2d with circular padding (canonical Lenia)."""
        pad = self.kernel_radius
        s4d = state_2d.unsqueeze(0).unsqueeze(0)
        padded = F.pad(s4d, (pad, pad, pad, pad), mode="circular")
        return F.conv2d(padded, self.kernel).squeeze(0).squeeze(0)

    def _excitation_spatial_batched(self, states: torch.Tensor) -> torch.Tensor:
        """Batched spatial excitation: states [B, H, W] → [B, H, W]."""
        pad = self.kernel_radius
        s4d = states.unsqueeze(1)  # [B, 1, H, W]
        padded = F.pad(s4d, (pad, pad, pad, pad), mode="circular")
        return F.conv2d(padded, self.kernel).squeeze(1)  # [B, H, W]

    @torch.no_grad()
    def decompose(
        self, board: Board, *,
        blind_mask: torch.Tensor | None = None,
        salience_map: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (excitation, growth, update) without modifying board state.

        Exposes the creature's sensory→reaction pipeline:
          excitation = conv(state, kernel)   — what each cell "senses"
          growth     = growth_fn(excitation) — the cell's response (+grow / -decay)
          update     = dt * growth           — actual state change per tick

        When salience_map is provided, uses renormalized convolution with W
        directly (Eq. 3).  When blind_mask is provided, uses renormalized
        convolution with V = 1 - M (Eq. 2).  salience_map takes precedence.
        """
        current = board.tensor
        H, W_dim = current.shape
        if salience_map is not None:
            # Generalized salience field (Eq. 3): W >= 0, no inversion
            kfft = self._rebuild_kernel_fft(H, W_dim)
            num = irfft2(rfft2(current * salience_map) * kfft, s=(H, W_dim))
            den = irfft2(rfft2(salience_map) * kfft, s=(H, W_dim))
            excitation = num / den.clamp(min=1e-6)
        elif blind_mask is not None:
            kfft = self._rebuild_kernel_fft(H, W_dim)
            visible = 1 - blind_mask
            exc_fft = rfft2(current * visible) * kfft
            vis_fft = rfft2(visible) * kfft
            excitation = irfft2(exc_fft, s=(H, W_dim)) / irfft2(vis_fft, s=(H, W_dim)).clamp(min=1e-6)
        elif self.use_fft:
            kfft = self._rebuild_kernel_fft(H, W_dim)
            excitation = irfft2(rfft2(current) * kfft, s=(H, W_dim))
        else:
            excitation = self._excitation_spatial(current)
        growth = _growth(excitation, self._mu_t, self._sigma_safe, self.cfg.growth_type)
        update = self.cfg.dt * growth
        return excitation, growth, update

    @torch.no_grad()
    def step(
        self, board: Board, *,
        blind_mask: torch.Tensor | None = None,
        salience_map: torch.Tensor | None = None,
    ) -> None:
        """Step the automaton forward one tick.

        Default: spatial F.conv2d (canonical Lenia).
        Opt-in: FFT convolution via rfft2 multiply (use_fft=True).

        salience_map takes precedence over blind_mask.  When salience_map is
        provided, uses renormalized convolution with W directly (Eq. 3).
        When blind_mask is provided, uses V = 1 - M (Eq. 2).
        """
        cfg = self.cfg
        current = board.tensor
        H, W_dim = current.shape

        if salience_map is not None:
            # Generalized salience field (Eq. 3): W >= 0, no inversion
            kfft = self._rebuild_kernel_fft(H, W_dim)
            num = irfft2(rfft2(current * salience_map) * kfft, s=(H, W_dim))
            den = irfft2(rfft2(salience_map) * kfft, s=(H, W_dim))
            excitation = num / den.clamp(min=1e-6)
        elif blind_mask is not None:
            # Renormalized convolution always uses FFT (two convolutions needed)
            kfft = self._rebuild_kernel_fft(H, W_dim)
            visible = 1 - blind_mask
            masked_current = current * visible
            exc_fft = rfft2(masked_current) * kfft
            vis_fft = rfft2(visible) * kfft
            excitation = irfft2(exc_fft, s=(H, W_dim))
            vis_weight = irfft2(vis_fft, s=(H, W_dim))
            excitation = excitation / vis_weight.clamp(min=1e-6)
        elif self.use_fft:
            kfft = self._rebuild_kernel_fft(H, W_dim)
            excitation = irfft2(rfft2(current) * kfft, s=(H, W_dim))
        else:
            excitation = self._excitation_spatial(current)

        growth = _growth(excitation, self._mu_t, self._sigma_safe, cfg.growth_type)
        if not isinstance(growth, torch.Tensor):
            growth = torch.as_tensor(growth, dtype=cfg.dtype, device=cfg.device)

        updated = torch.clamp(current + cfg.dt * growth, 0.0, 1.0)
        current.copy_(updated)

    @torch.no_grad()
    def step_batched(
        self,
        states: torch.Tensor,
        *,
        blind_masks: torch.Tensor | None = None,
        vis_weight: torch.Tensor | None = None,
        salience_maps: torch.Tensor | None = None,
        sal_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Step B simulations forward in parallel.

        Default: spatial F.conv2d (canonical Lenia).
        Opt-in: FFT convolution (use_fft=True).

        vis_weight: pre-computed renormalization denominator for persistent
        blind masks.  When provided the two FFTs for the visibility field
        are skipped, saving ~2× [B,H,W] peak memory per step.

        salience_maps: generalized salience field W >= 0 (Eq. 3).  Takes
        precedence over blind_masks.  sal_weight is the pre-computed
        denominator (analogous to vis_weight).
        """
        cfg = self.cfg
        H, W_dim = states.shape[1], states.shape[2]

        if salience_maps is not None:
            # Generalized salience field (Eq. 3)
            kfft = self._rebuild_kernel_fft(H, W_dim)
            if salience_maps.dim() == 2:
                salience_maps = salience_maps.unsqueeze(0).expand_as(states)
            num = irfft2(rfft2(states * salience_maps) * kfft, s=(H, W_dim))
            if sal_weight is None:
                sal_weight = irfft2(rfft2(salience_maps) * kfft, s=(H, W_dim))
            excitation = num / sal_weight.clamp(min=1e-6)
        elif blind_masks is not None:
            # Renormalized convolution always uses FFT
            kfft = self._rebuild_kernel_fft(H, W_dim)
            if blind_masks.dim() == 2:
                blind_masks = blind_masks.unsqueeze(0).expand_as(states)
            masked = states * (1 - blind_masks)
            exc_fft = rfft2(masked) * kfft          # broadcasts [B,H,W//2+1]
            del masked
            excitation = irfft2(exc_fft, s=(H, W_dim))
            del exc_fft
            if vis_weight is None:
                # Fallback: compute on the fly (mixed transient+persistent masks)
                visible = 1 - blind_masks
                vis_fft = rfft2(visible) * kfft
                del visible
                vis_weight = irfft2(vis_fft, s=(H, W_dim))
                del vis_fft
            excitation = excitation / vis_weight.clamp(min=1e-6)
        elif self.use_fft:
            kfft = self._rebuild_kernel_fft(H, W_dim)
            excitation = irfft2(rfft2(states) * kfft, s=(H, W_dim))  # [B, H, W]
        else:
            excitation = self._excitation_spatial_batched(states)

        growth = _growth(excitation, self._mu_t, self._sigma_safe, cfg.growth_type)
        updated = torch.clamp(states + cfg.dt * growth, 0.0, 1.0)
        return updated


class Lenia:
    def __init__(self, board: Board, automaton: Automaton):
        self.board = board
        self.automaton = automaton
        self.tick = 0

    @classmethod
    def from_config(cls, cfg: Config, *, fft: bool = True) -> "Lenia":
        """Construct a Lenia instance by instantiating board and automaton from Config."""
        board = Board(cfg)
        automaton = Automaton(cfg, fft=fft)
        return cls(board, automaton)

    def step(
        self, *,
        blind_mask: torch.Tensor | None = None,
        salience_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Step the simulation forward, optionally with blindness or salience."""
        self.automaton.step(self.board, blind_mask=blind_mask, salience_map=salience_map)
        self.tick += 1
        return self.board.tensor

    @property
    def config(self) -> Config:
        return self.board.cfg


def _make_kernel(radius: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    ax = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    dist2 = xx ** 2 + yy ** 2
    sigma = max(radius / 2.0, 0.5)
    kernel = torch.exp(-dist2 / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel.to(dtype=dtype, device=device)


def _growth(n: torch.Tensor, m: torch.Tensor | float, s: float, gtype: int) -> torch.Tensor:
    """Growth function mapping excitation to update magnitude."""
    if isinstance(m, torch.Tensor):
        m_val = m
    else:
        m_val = torch.tensor(m, dtype=n.dtype, device=n.device)
    s_safe = max(s, 1e-6) if isinstance(s, (int, float)) else s
    if gtype == 1:
        denom = max(9 * s_safe ** 2, 1e-6)
        base = 1 - (n - m_val) ** 2 / denom
        return torch.clamp(base, min=0.0).pow(4) * 2 - 1
    if gtype == 3:
        mask = (n - m_val).abs() <= s_safe
        return mask.to(dtype=n.dtype) * 2 - 1
    denom = max(2 * s_safe ** 2, 1e-6)
    return torch.exp(-((n - m_val) ** 2) / denom) * 2 - 1


def build_kernel(
    shape: Sequence[int],
    *,
    radius: float,
    kn: int,
    params: dict[str, object],
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    target_device = device or _auto_device()
    target_dtype = dtype or torch.float32
    rows, cols = shape
    cx = (rows - 1) / 2.0
    cy = (cols - 1) / 2.0
    yy, xx = torch.meshgrid(
        torch.arange(rows, device=target_device, dtype=target_dtype),
        torch.arange(cols, device=target_device, dtype=target_dtype),
        indexing="ij",
    )
    dx = (xx - cy) / radius
    dy = (yy - cx) / radius
    dist = torch.sqrt(dx ** 2 + dy ** 2)
    kfunc = _kernel_core(kn)
    if "rings" in params:
        # Multi-ring kernel: each ring has center r, half-width w, amplitude b.
        # arg maps distance into [0,1] relative to the ring, then kfunc shapes it.
        rings = params["rings"]
        profile = torch.zeros_like(dist)
        for ring in rings:  # type: ignore[assignment]
            r = float(ring.get("r", 0.5))
            w = max(float(ring.get("w", 0.1)), 1e-6)
            b = float(ring.get("b", 0.0))
            # Map radial distance into ring-local [0,1] coordinate
            arg = (dist - r) / (2 * w) + 0.5
            profile += kfunc(arg) * b
        kernel = profile
    else:
        b_values = params.get("b")
        if not b_values:
            kernel = kfunc(dist.clamp(0.0, 1.0))
        else:
            # Shell-based kernel: radius divided into B concentric shells,
            # each with amplitude from b_values. Br maps distance to shell index.
            r = float(params.get("r", 1.0))
            weights = torch.tensor(
                _fractions_to_float_list(b_values),
                dtype=target_dtype,
                device=target_device,
            )
            B = len(weights)
            Br = B * dist / max(r, 1e-6)
            idx = torch.minimum(torch.floor(Br).to(torch.int64), torch.tensor(B - 1, device=target_device))
            b = weights[idx]
            arg = torch.minimum(Br % 1.0, torch.tensor(1.0, device=target_device, dtype=target_dtype))
            shell = (dist < r).to(target_dtype)
            kernel = shell * kfunc(arg) * b
    return kernel.to(dtype=target_dtype, device=target_device)


def _kernel_core(kind: int):
    """Return the radial basis function for kernel construction.

    kn=0: smooth bump (exponential with compact support, exp(4 - 1/(r(1-r))))
    kn=1: polynomial bump ((4r(1-r))^4, most common in Lenia creatures)
    kn=3: flat-top / step function (plateau between 0.25 and 0.75)
    kn=4: Gaussian peak (narrow bell curve centered at r=0.5)
    """
    if kind == 1:  # polynomial bump
        return lambda r: ((r > 0) & (r < 1)).to(r.dtype) * (4 * r * (1 - r)) ** 4
    if kind == 3:  # flat-top step
        return lambda r, q=0.25: ((r >= q) & (r <= 1 - q)).to(r.dtype)
    if kind == 4:  # Gaussian peak
        return lambda r: ((r > 0) & (r < 1)).to(r.dtype) * torch.exp(-((r - 0.5) / 0.15) ** 2 / 2)
    # kn=0 (default): smooth bump with exponential compact support
    return lambda r: ((r > 0) & (r < 1)).to(r.dtype) * torch.nan_to_num(torch.exp(4 - 1 / (r * (1 - r))), nan=0.0, posinf=0.0, neginf=0.0)


def _fractions_to_float_list(values: Iterable[Fraction | float | int | str]) -> list[float]:
    return [float(v) for v in values]


def _kernel_to_conv_weight(kernel: torch.Tensor | Sequence[Sequence[float]], cfg: Config) -> torch.Tensor:
    tensor = torch.as_tensor(kernel, dtype=cfg.dtype, device=cfg.device)
    while tensor.dim() < 4:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() != 4:
        raise ValueError("Kernel tensor must be 2D or 4D")
    total = tensor.sum()
    if total.abs() < 1e-10:
        raise ValueError("Kernel sums to zero — cannot normalize")
    tensor = tensor / total
    return tensor.contiguous()
