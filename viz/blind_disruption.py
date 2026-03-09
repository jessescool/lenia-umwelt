"""Visualize how blindness disrupts a Lenia creature's sensory pipeline.

Two parts:
  1. t=0 disruption map: creature at steady state, instant blindness imposed.
     Same state decomposed with full vision vs blind — the true sensory cost.
  2. Time series of the blinded creature's own sensory health as it copes
     (or fails). No counterfactual — just its internal vitals.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from substrate.animals import load_animals
from substrate.lenia import Config, Board, Automaton, _growth


def _bbox(mask, pad=4):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    H, W = mask.shape
    return (max(rmin - pad, 0), min(rmax + pad + 1, H),
            max(cmin - pad, 0), min(cmax + pad + 1, W))


def _make_blind_mask(shape, center_r, center_c, radius):
    H, W = shape
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32), indexing="ij")
    return (torch.sqrt((yy - center_r) ** 2 + (xx - center_c) ** 2) <= radius).float()


def _growth_np(E, mu, sigma, gtype):
    """Numpy growth function for the curve plot."""
    s = max(sigma, 1e-6)
    if gtype == 1:
        base = np.clip(1 - (E - mu) ** 2 / (9 * s ** 2), 0, None)
        return base ** 4 * 2 - 1
    if gtype == 3:
        return (np.abs(E - mu) <= s).astype(float) * 2 - 1
    return np.exp(-((E - mu) ** 2) / (2 * s ** 2)) * 2 - 1


def blind_disruption(
    code: str = "O2u",
    scale: int = 1,
    warmup_steps: int = 200,
    track_steps: int = 60,
    grid: int = 64,
    blind_radius: int = 4,
    blind_site: str = "peripheral",
    output: str | Path | None = None,
):
    catalog = ROOT / "animals.json"
    animal = load_animals(catalog, codes=[code])[0]
    params = animal.params
    R = int(float(params.get("R", 13)))
    mu = float(params.get("m", 0.15))
    sigma = float(params.get("s", 0.015))
    gtype = int(float(params.get("gn", 2)))

    cfg = Config.from_animal(animal, base_grid=grid, scale=scale)
    cfg.device = torch.device("cpu")

    board = Board(cfg)
    auto = Automaton(cfg)
    cells = animal.cells
    if scale > 1:
        cells = cells.repeat_interleave(scale, dim=0).repeat_interleave(scale, dim=1)
    board.place(cells)
    for _ in range(warmup_steps):
        auto.step(board)

    state0 = board.tensor.clone()
    nonzero = torch.nonzero((state0 > 0.01).float())
    centroid = nonzero.float().mean(dim=0)
    if blind_site == "center":
        br, bc = int(round(centroid[0].item())), int(round(centroid[1].item()))
    else:
        # Peripheral: 60th percentile distance from centroid
        dists = torch.norm(nonzero.float() - centroid, dim=1)
        target_idx = torch.argsort(dists)[int(0.6 * len(dists))]
        br, bc = nonzero[target_idx][0].item(), nonzero[target_idx][1].item()

    H, W = cfg.grid_shape
    blind_mask = _make_blind_mask((H, W), br, bc, blind_radius)

    # t=0 disruption map (same state, two decompositions)
    exc_full, gro_full, _ = auto.decompose(board)
    exc_blind, gro_blind, _ = auto.decompose(board, blind_mask=blind_mask)

    state_np = state0.numpy()
    body = state_np > 0.01
    r0, r1, c0, c1 = _bbox(body, pad=5)
    body_c = body[r0:r1, c0:c1]

    bm_c = blind_mask.numpy()[r0:r1, c0:c1]
    byx = np.argwhere(bm_c > 0.5)
    br_c, bc_c = int(byx[:, 0].mean()), int(byx[:, 1].mean())

    def crop(t): return t.numpy()[r0:r1, c0:c1] if isinstance(t, torch.Tensor) else t[r0:r1, c0:c1]
    def mask_bg(arr):
        out = arr.astype(float).copy()
        out[~body_c] = np.nan
        return out

    exc_full_c = crop(exc_full)
    exc_blind_c = crop(exc_blind)
    gro_full_c = crop(gro_full)
    gro_blind_c = crop(gro_blind)
    exc_diff_c = exc_blind_c - exc_full_c   # signed
    gro_diff_c = gro_blind_c - gro_full_c
    state_c = crop(state_np)

    # Run blinded creature, track its own vitals
    board_blind = Board(cfg)
    board_blind.cells = state0.clone()
    auto_blind = Automaton(cfg)

    vitals = []
    for t in range(track_steps + 1):
        s = board_blind.tensor
        body_t = (s > 0.01)
        n_body = int(body_t.sum())
        if n_body == 0:
            vitals.append(dict(t=t, mass=0, exc_mean=0, exc_std=0,
                               gro_pos_frac=0, gro_neg_frac=0, gro_mean=0))
        else:
            exc_t, gro_t, _ = auto_blind.decompose(board_blind, blind_mask=blind_mask)
            eb = exc_t[body_t]
            gb = gro_t[body_t]
            vitals.append(dict(
                t=t,
                mass=float(s[body_t].sum()),
                exc_mean=float(eb.mean()),
                exc_std=float(eb.std()),
                gro_pos_frac=float((gb > 0).sum()) / n_body,
                gro_neg_frac=float((gb < 0).sum()) / n_body,
                gro_mean=float(gb.mean()),
            ))
        if t < track_steps:
            auto_blind.step(board_blind, blind_mask=blind_mask)

    time_ax = np.array([v["t"] for v in vitals])
    mass = np.array([v["mass"] for v in vitals])
    exc_mean = np.array([v["exc_mean"] for v in vitals])
    exc_std = np.array([v["exc_std"] for v in vitals])
    gro_pos = np.array([v["gro_pos_frac"] for v in vitals])
    gro_neg = np.array([v["gro_neg_frac"] for v in vitals])
    gro_mean = np.array([v["gro_mean"] for v in vitals])

    BG = "#1a1a2e"
    fig = plt.figure(figsize=(22, 16), facecolor=BG)
    gs = fig.add_gridspec(2, 1, top=0.92, bottom=0.05, left=0.04, right=0.96,
                          hspace=0.3, height_ratios=[1, 1])

    gs_top = gs[0].subgridspec(2, 4, hspace=0.25, wspace=0.25)

    # Excitation color range centered on mu
    exc_all = np.concatenate([exc_full_c[body_c], exc_blind_c[body_c]])
    exc_spread = max(abs(exc_all.min() - mu), abs(exc_all.max() - mu), 4 * sigma)
    ediff_lim = max(abs(exc_diff_c[body_c].min()), abs(exc_diff_c[body_c].max()), 1e-6)
    gdiff_lim = max(abs(gro_diff_c[body_c].min()), abs(gro_diff_c[body_c].max()), 1e-6)

    def _circle(ax):
        ax.add_patch(Circle((bc_c, br_c), blind_radius, lw=1.5,
                            ec="#00ffcc", fc="none", ls="--", alpha=0.9))

    panels_top = [
        (0, 0, "State", mask_bg(state_c), "inferno", 0, 1, False),
        (0, 1, "Excitation (full vision)", mask_bg(exc_full_c), "coolwarm",
         mu - exc_spread, mu + exc_spread, False),
        (0, 2, "Excitation (blinded)", mask_bg(exc_blind_c), "coolwarm",
         mu - exc_spread, mu + exc_spread, True),
        (0, 3, "E(blind) - E(full)", mask_bg(exc_diff_c), "PuOr_r",
         -ediff_lim, ediff_lim, True),
        (1, 0, "Growth (full vision)", mask_bg(gro_full_c), "RdBu_r", -1, 1, False),
        (1, 1, "Growth (blinded)", mask_bg(gro_blind_c), "RdBu_r", -1, 1, True),
        (1, 2, "G(blind) - G(full)", mask_bg(gro_diff_c), "PuOr_r",
         -gdiff_lim, gdiff_lim, True),
    ]

    for row, col, title, data, cmap, vmin, vmax, show_blind in panels_top:
        ax = fig.add_subplot(gs_top[row, col])
        cm = plt.get_cmap(cmap).copy()
        cm.set_bad(color=BG)
        ax.imshow(data, cmap=cm, vmin=vmin, vmax=vmax,
                  interpolation="nearest", origin="upper")
        if show_blind:
            _circle(ax)
        ax.set_title(title, color="white", fontsize=10, fontweight="bold", pad=4)
        ax.set_xticks([]); ax.set_yticks([])

    # Growth function + histograms in bottom-right of top section
    ax_gf = fig.add_subplot(gs_top[1, 3])
    E_range = np.linspace(0, 0.3, 500)
    G_curve = _growth_np(E_range, mu, sigma, gtype)
    ax_gf.plot(E_range, G_curve, color="white", linewidth=2)
    ax_gf.axhline(0, color="#aaaacc", linewidth=0.8, linestyle=":")
    ax_gf.axvline(mu, color="#ffcc00", linewidth=1.2, linestyle="--")
    ax_h = ax_gf.twinx()
    ax_h.hist(exc_full_c[body_c], bins=30, color="#4ecdc4", alpha=0.4,
              density=True, edgecolor="none", label="full")
    ax_h.hist(exc_blind_c[body_c], bins=30, color="#ff6b6b", alpha=0.4,
              density=True, edgecolor="none", label="blind")
    ax_h.set_ylabel("density", color="white", fontsize=8)
    ax_h.tick_params(colors="white", labelsize=7)
    ax_h.legend(fontsize=7, facecolor="#2a2a4a", edgecolor="white", labelcolor="white",
                loc="upper right")
    ax_gf.set_title("Growth fn + E distributions", color="white", fontsize=10,
                    fontweight="bold", pad=4)
    ax_gf.set_xlabel("excitation", color="white", fontsize=9)
    ax_gf.set_ylabel("growth", color="white", fontsize=9)
    ax_gf.set_xlim(0, 0.3)
    ax_gf.set_ylim(-1.1, 1.3)
    _style_ax(ax_gf, BG)

    sign_flips = np.sign(gro_blind_c[body_c]) != np.sign(gro_full_c[body_c])
    flip_pct = sign_flips.mean() * 100
    fig.text(0.5, 0.93,
             f"t=0 Sensory Disruption: {code}  |  blind r={blind_radius}, R={R}  |  "
             f"{flip_pct:.1f}% of body cells have flipped grow/decay decisions",
             ha="center", va="bottom", color="white", fontsize=13, fontweight="bold")

    gs_bot = gs[1].subgridspec(1, 4, wspace=0.3)

    ax_m = fig.add_subplot(gs_bot[0])
    ax_m.plot(time_ax, mass, color="#4ecdc4", linewidth=2)
    ax_m.set_title("Total mass", color="white", fontsize=11, fontweight="bold")
    ax_m.set_xlabel("timestep", color="white", fontsize=10)
    ax_m.set_ylabel("sum of body cell values", color="white", fontsize=10)
    _style_ax(ax_m, BG)

    ax_e = fig.add_subplot(gs_bot[1])
    ax_e.fill_between(time_ax, exc_mean - exc_std, exc_mean + exc_std,
                      color="#ff6b6b", alpha=0.2)
    ax_e.plot(time_ax, exc_mean, color="#ff6b6b", linewidth=2)
    ax_e.axhline(mu, color="#ffcc00", linewidth=1, linestyle=":", label=f"mu={mu}")
    ax_e.set_title("Excitation on body", color="white", fontsize=11, fontweight="bold")
    ax_e.set_xlabel("timestep", color="white", fontsize=10)
    ax_e.set_ylabel("mean +/- std", color="white", fontsize=10)
    ax_e.legend(fontsize=8, facecolor="#2a2a4a", edgecolor="white", labelcolor="white")
    _style_ax(ax_e, BG)

    ax_g = fig.add_subplot(gs_bot[2])
    ax_g.plot(time_ax, gro_pos * 100, color="#4ecdc4", linewidth=2, label="growing")
    ax_g.plot(time_ax, gro_neg * 100, color="#ff6b6b", linewidth=2, label="decaying")
    ax_g.axhline(50, color="#aaaacc", linewidth=0.8, linestyle=":")
    ax_g.set_title("Growth balance", color="white", fontsize=11, fontweight="bold")
    ax_g.set_xlabel("timestep", color="white", fontsize=10)
    ax_g.set_ylabel("% of body cells", color="white", fontsize=10)
    ax_g.set_ylim(0, 100)
    ax_g.legend(fontsize=8, facecolor="#2a2a4a", edgecolor="white", labelcolor="white")
    _style_ax(ax_g, BG)

    ax_n = fig.add_subplot(gs_bot[3])
    ax_n.fill_between(time_ax, 0, gro_mean, where=gro_mean >= 0,
                      color="#4ecdc4", alpha=0.3)
    ax_n.fill_between(time_ax, 0, gro_mean, where=gro_mean < 0,
                      color="#ff6b6b", alpha=0.3)
    ax_n.plot(time_ax, gro_mean, color="white", linewidth=2)
    ax_n.axhline(0, color="#aaaacc", linewidth=0.8, linestyle=":")
    ax_n.set_title("Net growth (mean)", color="white", fontsize=11, fontweight="bold")
    ax_n.set_xlabel("timestep", color="white", fontsize=10)
    ax_n.set_ylabel("mean growth on body", color="white", fontsize=10)
    _style_ax(ax_n, BG)

    site_tag = "center" if blind_site == "center" else "periph"
    out = output or ROOT / "results" / "new" / f"{code}_blind_{site_tag}_r{blind_radius}.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved → {out}")


def _style_ax(ax, bg):
    ax.tick_params(colors="white", labelsize=8)
    ax.set_facecolor(bg)
    ax.spines["bottom"].set_color("white")
    ax.spines["left"].set_color("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("code", nargs="?", default="O2u")
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--blind-radius", type=int, default=4)
    parser.add_argument("--blind-site", choices=["center", "peripheral"], default="peripheral")
    args = parser.parse_args()
    blind_disruption(
        args.code, scale=args.scale, warmup_steps=args.warmup,
        track_steps=args.steps, grid=args.grid,
        blind_radius=args.blind_radius, blind_site=args.blind_site,
    )
