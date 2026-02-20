import pandas as pd
import numpy as np

def cap_rows_per_group_binned(
    X_sub: pd.DataFrame,
    y_sub,
    g_sub,
    x_col: str = "Energy",   # already log(E)
    max_rows: int = 3000,
    n_bins: int = 200,
    seed: int = 0,
):
    """
    Cap each group (g_sub) to at most max_rows by sampling evenly across bins of X_sub[x_col].
    Assumes x_col is already log-energy.
    """
    rng = np.random.default_rng(seed)

    if x_col not in X_sub.columns:
        raise KeyError(f"'{x_col}' not found in X_sub columns.")

    y_arr = np.asarray(y_sub)
    g_arr = np.asarray(g_sub)

    pos = np.arange(len(X_sub))
    tmp = pd.DataFrame({"pos": pos, "g": g_arr})

    keep_pos_all = []

    for _, block in tmp.groupby("g", sort=False):
        block_pos = block["pos"].to_numpy()
        n = block_pos.size
        if n <= max_rows:
            keep_pos_all.append(block_pos)
            continue

        x = X_sub.iloc[block_pos][x_col].to_numpy()
        ok = np.isfinite(x)
        if ok.sum() < 2:
            keep_pos_all.append(rng.choice(block_pos, size=max_rows, replace=False))
            continue

        lo = np.nanmin(x[ok])
        hi = np.nanmax(x[ok])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            keep_pos_all.append(rng.choice(block_pos, size=max_rows, replace=False))
            continue

        edges = np.linspace(lo, hi, n_bins + 1)
        bin_id = np.digitize(x, edges) - 1
        bin_id = np.clip(bin_id, 0, n_bins - 1)

        target_per_bin = max_rows // n_bins
        remainder = max_rows - target_per_bin * n_bins

        keep_pos = []
        for b in range(n_bins):
            in_bin = block_pos[bin_id == b]
            if in_bin.size == 0:
                continue
            take = target_per_bin + (1 if b < remainder else 0)
            if in_bin.size <= take:
                keep_pos.append(in_bin)
            else:
                keep_pos.append(rng.choice(in_bin, size=take, replace=False))

        keep_pos = np.concatenate(keep_pos) if keep_pos else rng.choice(block_pos, size=max_rows, replace=False)

        # top up if we ended short due to empty bins
        if keep_pos.size < max_rows:
            remaining = np.setdiff1d(block_pos, keep_pos, assume_unique=False)
            if remaining.size > 0:
                add = rng.choice(remaining, size=min(max_rows - keep_pos.size, remaining.size), replace=False)
                keep_pos = np.concatenate([keep_pos, add])

        # trim if needed
        if keep_pos.size > max_rows:
            keep_pos = rng.choice(keep_pos, size=max_rows, replace=False)

        keep_pos_all.append(keep_pos)

    keep_pos_all = np.concatenate(keep_pos_all)
    keep_pos_all = np.sort(keep_pos_all)  # keep original order

    X_cap = X_sub.iloc[keep_pos_all]
    y_cap = y_arr[keep_pos_all]
    g_cap = g_arr[keep_pos_all]
    return X_cap, y_cap, g_cap

def filter_small_groups(X_sub, y_sub, g_sub, min_size):
    uniq, counts = np.unique(g_sub, return_counts=True)
    keep_groups = set(uniq[counts >= min_size])
    keep_mask = np.fromiter((gi in keep_groups for gi in g_sub), dtype=bool)
    return X_sub.loc[keep_mask], np.asarray(y_sub)[keep_mask], g_sub[keep_mask]