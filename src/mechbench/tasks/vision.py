"""Vision-modality mechanistic tasks.

Each task encodes a 2D image as a raster-flattened token grid so it slots
into the existing next-token-prediction framework. The probes isolate
primitives that 1D sequence models have to reconstruct to succeed on
vision-like inputs:

    grid_retrieval  — address-by-coordinate lookup after SEP. Tests whether
                      the model can learn a (row, col) -> position mapping
                      on a raster-flattened grid.

    col_parity      — XOR of a queried column in a binary grid. Column
                      elements are stride-W apart, so this probes the
                      ability to aggregate over non-adjacent positions
                      (trivial for full attention, costly for local-state
                      architectures).

    patch_match     — detect whether a 2x2 patch is duplicated somewhere
                      in the grid. Probes long-range 2D content matching
                      through a 1D-flattened view.
"""

from __future__ import annotations

import numpy as np

from .base import CONTENT_LO, PAD, SEP, TaskSpec, sample_content


def _empty(batch: int, T: int) -> tuple[np.ndarray, np.ndarray]:
    from .base import BOS
    records = np.full((batch, T + 1), PAD, dtype=np.int32)
    mask = np.zeros((batch, T), dtype=bool)
    records[:, 0] = BOS
    return records, mask


def _grid_side(T: int, overhead: int, cap: int = 10) -> int:
    """Pick the largest square side whose grid + overhead fits in T."""
    side = int(np.floor(np.sqrt(max(1, T - overhead))))
    return max(3, min(side, cap))


def grid_retrieval(rng, batch, spec: TaskSpec):
    """Coordinate-addressed retrieval in a raster-flattened 2D grid.

    Layout: [BOS, grid row-major (side*side tokens), SEP, row_tok, col_tok, value].
    Coord tokens come from the first ``side`` content-token ids; grid
    values are drawn from the rest of the content range so the two
    alphabets are disjoint and the model cannot confuse them.
    """
    T = spec.seq_len
    side = _grid_side(T, overhead=4, cap=10)
    # ensure coord alphabet and value alphabet both fit in content range
    if spec.n_content < side + 2:
        raise ValueError("vocab_size too small for grid_retrieval")
    H = W = side
    grid_size = H * W
    recs, mask = _empty(batch, T)
    coord_lo = CONTENT_LO
    val_lo = CONTENT_LO + side
    val_hi = spec.vocab_size
    idx = np.arange(batch)
    grid = rng.integers(val_lo, val_hi, size=(batch, H, W), dtype=np.int32)
    recs[:, 1 : 1 + grid_size] = grid.reshape(batch, grid_size)
    sep_pos = 1 + grid_size
    recs[:, sep_pos] = SEP
    r = rng.integers(0, H, size=batch, dtype=np.int64)
    c = rng.integers(0, W, size=batch, dtype=np.int64)
    recs[idx, sep_pos + 1] = coord_lo + r.astype(np.int32)
    recs[idx, sep_pos + 2] = coord_lo + c.astype(np.int32)
    recs[idx, sep_pos + 3] = grid[idx, r, c]
    mask[:, sep_pos + 2] = True
    return recs, mask


def col_parity(rng, batch, spec: TaskSpec):
    """XOR parity of a queried column in a binary 2D grid.

    Binary cells use content tokens ``CONTENT_LO`` (0) and ``CONTENT_LO+1``
    (1); the queried column index is encoded using a dedicated coord
    alphabet starting at ``CONTENT_LO + 2``.
    """
    T = spec.seq_len
    side = _grid_side(T, overhead=4, cap=10)
    if spec.n_content < 2 + side:
        raise ValueError("vocab_size too small for col_parity")
    H = W = side
    grid_size = H * W
    recs, mask = _empty(batch, T)
    zero = CONTENT_LO
    one = CONTENT_LO + 1
    coord_lo = CONTENT_LO + 2
    idx = np.arange(batch)
    bits = rng.integers(0, 2, size=(batch, H, W), dtype=np.int32)
    flat = np.where(bits.reshape(batch, grid_size) == 0, zero, one).astype(np.int32)
    recs[:, 1 : 1 + grid_size] = flat
    sep_pos = 1 + grid_size
    recs[:, sep_pos] = SEP
    c = rng.integers(0, W, size=batch, dtype=np.int64)
    recs[idx, sep_pos + 1] = coord_lo + c.astype(np.int32)
    parity = (bits[idx, :, c].sum(axis=1) % 2).astype(np.int32)
    recs[idx, sep_pos + 2] = zero + parity
    mask[:, sep_pos + 1] = True
    return recs, mask


def patch_match(rng, batch, spec: TaskSpec):
    """Detect a duplicated 2x2 patch in an otherwise random grid.

    Half the samples have a 2x2 patch planted at two non-overlapping
    locations; the other half are left random. After SEP the model emits
    a binary yes/no token. At side >= 6 with a content alphabet of 50+
    tokens, the probability of an incidental duplicate in the random case
    is < 1e-3, so the positive/negative signal is near-clean.
    """
    T = spec.seq_len
    side = _grid_side(T, overhead=3, cap=10)
    if side < 4:
        raise ValueError("grid too small for patch_match (need side >= 4)")
    H = W = side
    grid_size = H * W
    recs, mask = _empty(batch, T)
    no_tok = CONTENT_LO
    yes_tok = CONTENT_LO + 1
    # draw grids sample-by-sample so placements can depend on per-sample rng
    for b in range(batch):
        grid = sample_content(rng, spec, (H, W))
        duplicate = bool(rng.integers(0, 2))
        if duplicate:
            r1 = int(rng.integers(0, H - 1))
            c1 = int(rng.integers(0, W - 1))
            patch = grid[r1 : r1 + 2, c1 : c1 + 2].copy()
            # find a non-overlapping second location (bounding-box disjoint)
            for _ in range(50):
                r2 = int(rng.integers(0, H - 1))
                c2 = int(rng.integers(0, W - 1))
                if abs(r2 - r1) >= 2 or abs(c2 - c1) >= 2:
                    grid[r2 : r2 + 2, c2 : c2 + 2] = patch
                    break
            else:
                duplicate = False  # grid too small to place disjointly
        recs[b, 1 : 1 + grid_size] = grid.reshape(-1)
        sep_pos = 1 + grid_size
        recs[b, sep_pos] = SEP
        recs[b, sep_pos + 1] = yes_tok if duplicate else no_tok
        mask[b, sep_pos] = True
    return recs, mask
