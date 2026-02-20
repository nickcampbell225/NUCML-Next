import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

def make_actinide_nearby_splits(
    df_main: pd.DataFrame,
    X: pd.DataFrame,
    y,
    *,
    Z0: int = 92,
    A0: int = 235,
    act_Z_min: int = 89,
    act_Z_max: int = 103,
    K_nuclides: int = 50,
    n_outer_nuclides: int = 15,
    wZ: float = 10.0,
    wA: float = 1.0,
    seed: int = 0,
    require_actinides_in_nearby: bool = True,
) -> Tuple[
    pd.DataFrame, np.ndarray, np.ndarray,
    pd.DataFrame, np.ndarray, np.ndarray,
    Dict[str, Any]
]:
    """
    Build tuning/outer splits by picking nearby actinide nuclides (Z,A) around (Z0,A0),
    then splitting those nuclides into outer vs tuning sets, while LOGO groups are (Z,A,MT).

    Returns:
      X_tune, y_tune, g_tune, X_outer, y_outer, g_outer, info_dict
    """
    # --- basic checks ---
    needed_cols = {"Z", "A", "MT"}
    missing = needed_cols - set(df_main.columns)
    if missing:
        raise ValueError(f"df_main is missing columns: {sorted(missing)}")

    if len(df_main) != len(X) or len(df_main) != len(y):
        raise ValueError("df_main, X, and y must have the same number of rows")

    y_arr = np.asarray(y)

    # --- 1) restrict df_main to actinides ---
    act_mask = df_main["Z"].between(act_Z_min, act_Z_max)
    df_act = df_main.loc[act_mask, ["Z", "A"]].copy()

    # --- 2) pick nearby NUCLIDES (Z,A) ---
    nuclides = df_act.drop_duplicates().copy()
    nuclides["d"] = np.sqrt((wZ * (nuclides["Z"] - Z0)) ** 2 + (wA * (nuclides["A"] - A0)) ** 2)

    if len(nuclides) == 0:
        raise ValueError("No nuclides found after actinide filtering; check act_Z_min/act_Z_max.")

    K_eff = min(K_nuclides, len(nuclides))
    near_nuclides_df = nuclides.sort_values("d").head(K_eff)[["Z", "A"]]

    near_pairs = set(map(tuple, near_nuclides_df.to_numpy()))
    nuclide_all = np.column_stack([df_main["Z"].to_numpy(), df_main["A"].to_numpy()])

    # vectorized membership is a bit awkward with tuples; this is fine for typical sizes
    near_row_mask = np.fromiter(((int(z), int(a)) in near_pairs for z, a in nuclide_all),
                                dtype=bool, count=len(df_main))

    if require_actinides_in_nearby:
        near_row_mask &= act_mask.to_numpy()

    # --- 3) split by NUCLIDE into tuning vs outer-validation nuclides ---
    near_labels = np.array(list(near_pairs), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(near_labels)

    n_outer_eff = min(n_outer_nuclides, len(near_labels))
    outer_nuclides = set(map(tuple, near_labels[:n_outer_eff]))
    tune_nuclides  = set(map(tuple, near_labels[n_outer_eff:]))

    outer_row_mask = np.fromiter(((int(z), int(a)) in outer_nuclides for z, a in nuclide_all),
                                 dtype=bool, count=len(df_main)) & near_row_mask
    tune_row_mask  = np.fromiter(((int(z), int(a)) in tune_nuclides  for z, a in nuclide_all),
                                 dtype=bool, count=len(df_main)) & near_row_mask

    # --- 4) LOGO groups are (Z,A,MT) ---
    groups_main = pd.factorize(pd.MultiIndex.from_frame(df_main[["Z", "A", "MT"]]))[0]

    X_tune  = X.loc[tune_row_mask]
    y_tune  = y_arr[tune_row_mask]
    g_tune  = groups_main[tune_row_mask]

    X_outer = X.loc[outer_row_mask]
    y_outer = y_arr[outer_row_mask]
    g_outer = groups_main[outer_row_mask]

    info = dict(
        act_mask=act_mask.to_numpy(),
        near_row_mask=near_row_mask,
        tune_row_mask=tune_row_mask,
        outer_row_mask=outer_row_mask,
        near_nuclides=near_nuclides_df.reset_index(drop=True),
        tune_nuclides=tune_nuclides,
        outer_nuclides=outer_nuclides,
        groups_main=groups_main,
        params=dict(
            Z0=Z0, A0=A0, act_Z_min=act_Z_min, act_Z_max=act_Z_max,
            K_nuclides=K_nuclides, n_outer_nuclides=n_outer_nuclides,
            wZ=wZ, wA=wA, seed=seed,
        ),
    )

    return X_tune, y_tune, g_tune, X_outer, y_outer, g_outer, info