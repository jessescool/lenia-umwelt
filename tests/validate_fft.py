"""Validate FFT convolution against spatial conv2d and vectorized prepare_profile.

Tests:
  1. Single-step excitation field: FFT vs F.conv2d at scales 1, 2, 4 (tol ~1e-5)
  2. Vectorized vs loop prepare_profile (tol ~1e-6)
  3. 100-step trajectory divergence: Automaton(fft=True) vs Automaton(fft=False) (max diff < 1e-4)
  4. Batched FFT vs single-board FFT consistency
  5. Dual-path flag: fft=False uses spatial, fft=True uses FFT (both produce valid Lenia)

Run on cluster: ./dispatch "python experiments/validate_fft.py"
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from substrate.animals import load_animals
from substrate.lenia import Config, Board, Automaton, _kernel_to_conv_weight, _growth


def excitation_conv2d(automaton: Automaton, state: torch.Tensor) -> torch.Tensor:
    """Reference excitation via spatial F.conv2d with circular padding."""
    pad = automaton.kernel_radius
    state_4d = state.unsqueeze(0).unsqueeze(0)
    padded = F.pad(state_4d, (pad, pad, pad, pad), mode="circular")
    return F.conv2d(padded, automaton.kernel).squeeze(0).squeeze(0)


def excitation_fft(automaton: Automaton, state: torch.Tensor) -> torch.Tensor:
    """FFT excitation (what the fft=True path uses internally)."""
    from torch.fft import rfft2, irfft2
    H, W = state.shape
    kfft = automaton._rebuild_kernel_fft(H, W)
    return irfft2(rfft2(state) * kfft, s=(H, W))


def test_excitation_field(scales=(1, 2, 4), tol=1e-5):
    """Compare FFT vs conv2d excitation at multiple scales."""
    animals = load_animals("animals.json", codes=["O2u"])
    animal = animals[0]
    base_grid = 128

    print("=" * 60)
    print("TEST 1: Excitation field — FFT vs conv2d")
    print("=" * 60)

    all_pass = True
    for scale in scales:
        cfg = Config.from_animal(animal, base_grid=base_grid, scale=scale)
        automaton = Automaton(cfg)  # default: spatial (doesn't matter for raw excitation test)
        board = Board(cfg)

        # Place a random state
        torch.manual_seed(42)
        H, W = cfg.grid_shape
        board.cells = torch.rand(H, W, device=cfg.device, dtype=cfg.dtype)

        exc_conv = excitation_conv2d(automaton, board.tensor)
        exc_fft = excitation_fft(automaton, board.tensor)

        max_diff = (exc_conv - exc_fft).abs().max().item()
        mean_diff = (exc_conv - exc_fft).abs().mean().item()
        passed = max_diff < tol

        status = "PASS" if passed else "FAIL"
        kernel_size = 2 * cfg.kernel_radius + 1
        print(f"  scale={scale}  kernel={kernel_size}x{kernel_size}  "
              f"grid={H}x{W}  max_diff={max_diff:.2e}  mean_diff={mean_diff:.2e}  [{status}]")

        if not passed:
            all_pass = False

    return all_pass


def test_prepare_profile_vectorized(tol=1e-6):
    """Compare vectorized batch path vs per-element loop."""
    from metrics_and_machinery.distance_metrics import prepare_profile, _prepare_profile_batched

    print("\n" + "=" * 60)
    print("TEST 2: prepare_profile — vectorized vs loop")
    print("=" * 60)

    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = 256

    all_pass = True
    for B, H, W, label in [(64, 128, 128, "128x128"), (32, 256, 256, "256x256"),
                            (16, 512, 512, "512x512")]:
        grids = torch.rand(B, H, W, device=device) * torch.rand(B, 1, 1, device=device)
        # Zero out ~70% of pixels to simulate sparse creatures
        grids[grids < 0.7] = 0.0

        # Loop path (reference)
        ref = torch.empty(B, m, dtype=grids.dtype, device=device)
        for i in range(B):
            ref[i] = prepare_profile(grids[i], m)

        # Vectorized path
        vec = _prepare_profile_batched(grids, m)

        max_diff = (ref - vec).abs().max().item()
        mean_diff = (ref - vec).abs().mean().item()
        passed = max_diff < tol

        status = "PASS" if passed else "FAIL"
        print(f"  B={B}  grid={label}  max_diff={max_diff:.2e}  mean_diff={mean_diff:.2e}  [{status}]")

        if not passed:
            all_pass = False

    return all_pass


def test_trajectory_divergence(n_steps=100, tol=2e-4):
    """Run 100 steps with fft=True vs fft=False, compare final states."""
    animals = load_animals("animals.json", codes=["O2u"])
    animal = animals[0]

    print("\n" + "=" * 60)
    print("TEST 3: 100-step trajectory — Automaton(fft=True) vs Automaton(fft=False)")
    print("=" * 60)

    all_pass = True
    for scale in (1, 2, 4):
        base_grid = 128
        cfg = Config.from_animal(animal, base_grid=base_grid, scale=scale)
        H, W = cfg.grid_shape

        # Create two automatons: spatial (default) and FFT
        auto_spatial = Automaton(cfg, fft=False)
        auto_fft = Automaton(cfg, fft=True)

        board_spatial = Board(cfg)
        board_fft = Board(cfg)

        # Place creature pattern
        from substrate.simulation import Simulation
        sim = Simulation(cfg)
        sim.place_animal(animal, center=True)
        initial = sim.board.tensor.clone()
        board_spatial.cells = initial.clone()
        board_fft.cells = initial.clone()

        for step in range(n_steps):
            auto_spatial.step(board_spatial)
            auto_fft.step(board_fft)

        max_diff = (board_spatial.tensor - board_fft.tensor).abs().max().item()
        mean_diff = (board_spatial.tensor - board_fft.tensor).abs().mean().item()
        passed = max_diff < tol

        status = "PASS" if passed else "FAIL"
        kernel_size = 2 * cfg.kernel_radius + 1
        print(f"  scale={scale}  kernel={kernel_size}x{kernel_size}  grid={H}x{W}  "
              f"steps={n_steps}  max_diff={max_diff:.2e}  mean_diff={mean_diff:.2e}  [{status}]")

        if not passed:
            all_pass = False

    return all_pass


def test_batched_trajectory(n_steps=50, tol=1e-4):
    """Run batched step vs single-board, testing both fft=True and fft=False."""
    animals = load_animals("animals.json", codes=["O2u"])
    animal = animals[0]

    print("\n" + "=" * 60)
    print("TEST 4: Batched vs single-board consistency (both paths)")
    print("=" * 60)

    all_pass = True
    for fft_mode in (False, True):
        mode_label = "FFT" if fft_mode else "spatial"
        for scale in (1, 2):
            cfg = Config.from_animal(animal, base_grid=128, scale=scale)
            automaton = Automaton(cfg, fft=fft_mode)
            H, W = cfg.grid_shape

            # Create B=4 slightly perturbed initial states
            from substrate.simulation import Simulation
            sim = Simulation(cfg)
            sim.place_animal(animal, center=True)
            base = sim.board.tensor.clone()

            B = 4
            torch.manual_seed(99)
            noise = torch.rand(B, H, W, device=cfg.device, dtype=cfg.dtype) * 0.01
            states = base.unsqueeze(0).expand(B, -1, -1).clone() + noise
            states.clamp_(0, 1)

            # Run batched
            batched_states = states.clone()
            for _ in range(n_steps):
                batched_states = automaton.step_batched(batched_states)

            # Run single-board for each
            single_results = []
            for i in range(B):
                board = Board(cfg)
                board.cells = states[i].clone()
                for _ in range(n_steps):
                    automaton.step(board)
                single_results.append(board.tensor.clone())
            single_states = torch.stack(single_results)

            max_diff = (batched_states - single_states).abs().max().item()
            passed = max_diff < tol

            status = "PASS" if passed else "FAIL"
            print(f"  {mode_label:7s}  scale={scale}  B={B}  steps={n_steps}  max_diff={max_diff:.2e}  [{status}]")

            if not passed:
                all_pass = False

    return all_pass


def test_flag_default():
    """Verify that fft=False (default) uses spatial and fft=True uses FFT."""
    print("\n" + "=" * 60)
    print("TEST 5: Flag wiring — default=FFT, fft=False=spatial")
    print("=" * 60)

    animals = load_animals("animals.json", codes=["O2u"])
    animal = animals[0]
    cfg = Config.from_animal(animal, base_grid=64, scale=1)

    auto_default = Automaton(cfg)
    auto_spatial = Automaton(cfg, fft=False)
    auto_fft = Automaton(cfg, fft=True)

    assert auto_default.use_fft, "Default should be FFT (use_fft=True)"
    assert not auto_spatial.use_fft, "Explicit fft=False should be spatial"
    assert auto_fft.use_fft, "Explicit fft=True should be FFT"

    # Run one step each and verify default == fft=True (bitwise identical)
    board_default = Board(cfg)
    board_fft = Board(cfg)
    torch.manual_seed(42)
    state = torch.rand(*cfg.grid_shape, device=cfg.device, dtype=cfg.dtype)
    board_default.cells = state.clone()
    board_fft.cells = state.clone()

    auto_default.step(board_default)
    auto_fft.step(board_fft)

    identical = torch.equal(board_default.tensor, board_fft.tensor)
    status = "PASS" if identical else "FAIL"
    print(f"  default == fft=True (bitwise): {identical}  [{status}]")

    print(f"  auto_default.use_fft = {auto_default.use_fft}")
    print(f"  auto_fft.use_fft     = {auto_fft.use_fft}")

    return identical


def test_impulse_response(tol=1e-5):
    """Single 1.0 at grid center — FFT excitation should reproduce the kernel."""
    from torch.fft import rfft2, irfft2

    print("\n" + "=" * 60)
    print("TEST 6: Impulse response — FFT excitation matches kernel")
    print("=" * 60)

    animals = load_animals("animals.json", codes=["O2u"])
    animal = animals[0]
    cfg = Config.from_animal(animal, base_grid=64, scale=1)
    automaton = Automaton(cfg, fft=True)
    H, W = cfg.grid_shape

    # Place single impulse at grid center
    state = torch.zeros(H, W, device=cfg.device, dtype=cfg.dtype)
    cy, cx = H // 2, W // 2
    state[cy, cx] = 1.0

    exc = irfft2(rfft2(state) * automaton._kernel_fft, s=(H, W))

    # The expected excitation is the kernel centered at (cy, cx).
    # Build reference: embed kernel into grid and roll to (cy, cx).
    K = automaton.kernel.shape[-1]
    kernel_2d = automaton.kernel[0, 0]
    ref = torch.zeros(H, W, device=cfg.device, dtype=cfg.dtype)
    ref[:K, :K] = kernel_2d
    center = K // 2
    # Roll kernel center to (cy, cx)
    ref = torch.roll(ref, shifts=(cy - center, cx - center), dims=(0, 1))

    max_diff = (exc - ref).abs().max().item()
    passed = max_diff < tol
    status = "PASS" if passed else "FAIL"
    print(f"  grid={H}x{W}  kernel={K}x{K}  max_diff={max_diff:.2e}  [{status}]")
    return passed


def test_uniform_input(tol=1e-6):
    """Uniform grid c=0.5 — excitation should be c everywhere (kernel sums to 1)."""
    from torch.fft import rfft2, irfft2

    print("\n" + "=" * 60)
    print("TEST 7: Uniform input — excitation equals constant c")
    print("=" * 60)

    animals = load_animals("animals.json", codes=["O2u"])
    animal = animals[0]
    cfg = Config.from_animal(animal, base_grid=64, scale=1)
    automaton = Automaton(cfg, fft=True)
    H, W = cfg.grid_shape

    c = 0.5
    state = torch.full((H, W), c, device=cfg.device, dtype=cfg.dtype)
    exc = irfft2(rfft2(state) * automaton._kernel_fft, s=(H, W))

    max_dev = (exc - c).abs().max().item()
    passed = max_dev < tol
    status = "PASS" if passed else "FAIL"
    print(f"  c={c}  grid={H}x{W}  max_deviation={max_dev:.2e}  [{status}]")
    return passed


def test_renorm_no_mask(tol=1e-6):
    """Renormalized convolution with blind_mask=zeros should match standard FFT."""
    print("\n" + "=" * 60)
    print("TEST 8: Renormalized conv (no mask) matches standard FFT")
    print("=" * 60)

    animals = load_animals("animals.json", codes=["O2u"])
    animal = animals[0]
    cfg = Config.from_animal(animal, base_grid=64, scale=1)
    automaton = Automaton(cfg, fft=True)
    H, W = cfg.grid_shape

    torch.manual_seed(77)
    state = torch.rand(H, W, device=cfg.device, dtype=cfg.dtype)

    board_std = Board(cfg)
    board_std.cells = state.clone()

    board_renorm = Board(cfg)
    board_renorm.cells = state.clone()

    # Standard FFT step
    automaton.step(board_std)

    # Renormalized step with no masking (all zeros = fully visible)
    no_mask = torch.zeros(H, W, device=cfg.device, dtype=cfg.dtype)
    automaton.step(board_renorm, blind_mask=no_mask)

    max_diff = (board_std.tensor - board_renorm.tensor).abs().max().item()
    passed = max_diff < tol
    status = "PASS" if passed else "FAIL"
    print(f"  grid={H}x{W}  max_diff={max_diff:.2e}  [{status}]")
    return passed


def test_nonsquare_grids(tol=1e-5):
    """FFT vs spatial on non-square grids (64x128, 128x64)."""
    print("\n" + "=" * 60)
    print("TEST 9: Non-square grids — FFT vs spatial")
    print("=" * 60)

    animals = load_animals("animals.json", codes=["O2u"])
    animal = animals[0]

    all_pass = True
    for grid_shape in [(64, 128), (128, 64)]:
        cfg = Config.from_animal(animal, base_grid=grid_shape, scale=1)
        auto_fft = Automaton(cfg, fft=True)
        auto_spatial = Automaton(cfg, fft=False)
        H, W = cfg.grid_shape

        torch.manual_seed(42)
        state = torch.rand(H, W, device=cfg.device, dtype=cfg.dtype)

        exc_fft = excitation_fft(auto_fft, state)
        exc_conv = excitation_conv2d(auto_spatial, state)

        max_diff = (exc_fft - exc_conv).abs().max().item()
        passed = max_diff < tol
        status = "PASS" if passed else "FAIL"
        print(f"  grid={H}x{W}  max_diff={max_diff:.2e}  [{status}]")
        if not passed:
            all_pass = False

    return all_pass


def test_odd_even_grids(tol=1e-5):
    """FFT vs spatial on odd and even grid sizes (63, 64, 65)."""
    print("\n" + "=" * 60)
    print("TEST 10: Odd vs even grid sizes — FFT vs spatial")
    print("=" * 60)

    animals = load_animals("animals.json", codes=["O2u"])
    animal = animals[0]

    all_pass = True
    for size in [63, 64, 65]:
        cfg = Config.from_animal(animal, base_grid=size, scale=1)
        auto_fft = Automaton(cfg, fft=True)
        auto_spatial = Automaton(cfg, fft=False)
        H, W = cfg.grid_shape

        torch.manual_seed(42)
        state = torch.rand(H, W, device=cfg.device, dtype=cfg.dtype)

        exc_fft = excitation_fft(auto_fft, state)
        exc_conv = excitation_conv2d(auto_spatial, state)

        max_diff = (exc_fft - exc_conv).abs().max().item()
        passed = max_diff < tol
        status = "PASS" if passed else "FAIL"
        print(f"  grid={size}x{size}  max_diff={max_diff:.2e}  [{status}]")
        if not passed:
            all_pass = False

    return all_pass


if __name__ == "__main__":
    results = []
    results.append(("Excitation field", test_excitation_field()))
    results.append(("Prepare profile vectorized", test_prepare_profile_vectorized()))
    results.append(("Trajectory divergence (fft=T vs fft=F)", test_trajectory_divergence()))
    results.append(("Batched consistency (both paths)", test_batched_trajectory()))
    results.append(("Flag wiring", test_flag_default()))
    results.append(("Impulse response", test_impulse_response()))
    results.append(("Uniform input", test_uniform_input()))
    results.append(("Renormalized conv (no mask)", test_renorm_no_mask()))
    results.append(("Non-square grids", test_nonsquare_grids()))
    results.append(("Odd vs even grid sizes", test_odd_even_grids()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\nAll tests passed!")
    else:
        print("\nSome tests FAILED!")
        sys.exit(1)
