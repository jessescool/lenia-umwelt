"""Animal catalog helpers combined with Lenia RLE decoding primitives."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Sequence
import json

import torch

_PREFIX_CHARS = "pqrstuvwxy@"

@dataclass(slots=True)
class Animal:
    code: str
    name: str
    cname: str
    params: dict[str, object]
    cells: torch.Tensor


def decode_rle(payload: str) -> torch.Tensor:
    """Decode a Lenia-style RLE string into a float32 torch tensor."""
    rows: list[list[float]] = []
    row: list[float] = []
    count = ""
    prefix = ""
    stream = payload.strip()
    for ch in stream:
        if ch in "\n\r":
            continue
        if ch == '!':
            break
        if ch.isdigit():
            count += ch
            continue
        if ch in _PREFIX_CHARS:
            prefix = ch
            continue
        if ch == '$':
            run = int(count) if count else 1
            rows.append(row)
            row = []
            for _ in range(run - 1):
                rows.append([])
            count = ""
            prefix = ""
            continue
        token = '.' if ch == '.' else (prefix + ch if prefix else ch)
        value = symbol_to_value(token) / 255.0
        run = int(count) if count else 1
        if run <= 0:
            run = 1
        row.extend([value] * run)
        count = ""
        prefix = ""
    if row:
        rows.append(row)
    if not rows:
        return torch.zeros((0, 0), dtype=torch.float32)
    width = max(len(r) for r in rows)
    padded = [r + [0.0] * (width - len(r)) for r in rows]
    return torch.tensor(padded, dtype=torch.float32)


def decode_cells(rle: str) -> torch.Tensor:
    """Decode an animal's Lenia-RLE payload into a float32 torch tensor."""
    return decode_rle(rle)


def symbol_to_value(symbol: str) -> int:
    """Convert a Lenia RLE symbol to an integer brightness value (0-255).

    Single-char symbols: A=1, B=2, ..., X=24 (first 24 values).
    Two-char symbols: prefix 'p'-'y' (10 tiers) + suffix 'A'-'X' (24 per tier),
    starting at 25. So 'pA'=25, 'pB'=26, ..., 'pX'=48, 'qA'=49, etc.
    """
    if symbol in {'.', 'b', ''}:
        return 0
    if symbol == 'o':
        return 255
    if len(symbol) == 1:
        # Single char: A=1 through X=24
        return ord(symbol) - ord('A') + 1
    # Two-char: high tier ('p'=0, 'q'=1, ...) * 24 + low offset (A=25, B=26, ...)
    high = ord(symbol[0]) - ord('p')
    low = ord(symbol[1]) - ord('A') + 25
    return high * 24 + low


def fractions_from_string(text: str) -> list[Fraction]:
    parts = [part.strip() for part in text.strip('[]').split(',') if part.strip()]
    return [Fraction(part) for part in parts]


def load_animals(path: str | Path, *, codes: Sequence[str] | None = None) -> list[Animal]:
    """Load one or more animals from the canonical JSON catalog."""
    selected = set(codes) if codes is not None else None
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    animals: list[Animal] = []
    for entry in payload:
        code = entry.get("code", "")
        if not code or code.startswith(">"):
            continue
        if selected is not None and code not in selected:
            continue
        rle = entry.get("cells")
        if not rle:
            continue
        params = _parse_params(entry.get("params") or {})
        animals.append(
            Animal(
                code=code,
                name=entry.get("name", ""),
                cname=entry.get("cname", ""),
                params=params,
                cells=decode_cells(rle),
            )
        )
    return animals


def iter_animals(path: str | Path, codes: Iterable[str] | None = None) -> Iterable[Animal]:
    """Generator variant of load_animals for streaming large catalogs."""
    wanted = list(codes) if codes is not None else None
    for animal in load_animals(path, codes=wanted):
        yield animal


def _parse_params(raw: dict[str, object]) -> dict[str, object]:
    params = dict(raw)
    b_value = params.get("b")
    if isinstance(b_value, str):
        params["b"] = fractions_from_string(b_value)
    elif isinstance(b_value, list):
        params["b"] = [Fraction(str(item)) for item in b_value]
    return params
