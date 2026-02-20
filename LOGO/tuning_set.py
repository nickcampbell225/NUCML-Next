import numpy as np
import pandas as pd

def get_tuning_set(df: pd.DataFrame, df_holdout: pd.DataFrame, no_of_groups: int):
    """
    Select the `no_of_groups` (Z, A, MT) groups in `df` closest in (Z, A) to the single
    (Z, A, MT) group present in `df_holdout`.

    Safety: if the exact holdout (Z, A, MT) key exists in `df`, it is excluded.

    Returns:
        X_tune : DataFrame with all columns except ["Entry","MT","Uncertainty","CrossSection"]
        y_tune : DataFrame with only ["CrossSection"]
        g_tune : array-like group labels (length == len(X_tune)) suitable for LeaveOneGroupOut
    """
    required_cols = {"Z", "A", "MT", "CrossSection"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"`df` is missing required columns: {sorted(missing)}")

    missing_h = {"Z", "A", "MT"} - set(df_holdout.columns)
    if missing_h:
        raise ValueError(f"`df_holdout` is missing required columns: {sorted(missing_h)}")

    # Ensure holdout has exactly one unique (Z, A, MT) group
    holdout_groups = df_holdout[["Z", "A", "MT"]].drop_duplicates()
    if len(holdout_groups) != 1:
        raise ValueError(
            f"`df_holdout` must contain exactly 1 unique (Z, A, MT) group; found {len(holdout_groups)}."
        )
    hz, ha, hmt = holdout_groups.iloc[0].tolist()

    # Unique candidate groups from df
    group_keys = df[["Z", "A", "MT"]].drop_duplicates().copy()

    # Safety: exclude exact holdout group if present
    group_keys = group_keys[~((group_keys["Z"] == hz) & (group_keys["A"] == ha) & (group_keys["MT"] == hmt))]

    no_of_groups = int(no_of_groups)
    if no_of_groups <= 0:
        raise ValueError("`no_of_groups` must be a positive integer.")
    if len(group_keys) == 0:
        raise ValueError("No candidate (Z, A, MT) groups remain after excluding the holdout key.")
    if no_of_groups > len(group_keys):
        raise ValueError(
            f"Requested no_of_groups={no_of_groups}, but only {len(group_keys)} candidate groups are available."
        )

    # Distance in (Z, A) only
    group_keys["dist"] = np.sqrt((group_keys["Z"] - hz) ** 2 + (group_keys["A"] - ha) ** 2)
    nearest = group_keys.sort_values("dist", ascending=True).head(no_of_groups)

    # Keep all rows in df belonging to selected groups
    df_tune = df.merge(nearest[["Z", "A", "MT"]], on=["Z", "A", "MT"], how="inner")

    drop_cols = [c for c in ["Entry", "MT", "Uncertainty", "CrossSection"] if c in df_tune.columns]
    X_tune = df_tune.drop(columns=drop_cols)
    y_tune = df_tune[["CrossSection"]].copy()

    # Group labels per row for LeaveOneGroupOut
    g_tune = df_tune[["Z", "A", "MT"]].apply(tuple, axis=1).to_numpy()

    return X_tune, y_tune, g_tune