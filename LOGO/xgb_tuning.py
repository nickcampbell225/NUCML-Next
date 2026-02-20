import numpy as np

from sklearn.model_selection import LeaveOneGroupOut, RandomizedSearchCV

try:
    from xgboost import XGBRegressor
except ImportError as e:
    raise ImportError(
        "xgb_tuning requires xgboost. Install with `pip install xgboost`."
    ) from e


def xgb_tuning(
    X_tune,
    y_tune,
    g_tune,
    param_space: dict,
    base_kwargs: dict,
    *,
    n_iter: int = 50,
    scoring: str = "neg_root_mean_squared_error",
    random_state: int = 42,
    n_jobs: int = -1,
    verbose: int = 1,
    refit: bool = True,
):
    """
    Random search hyperparameter tuning for XGBRegressor using Leave-One-Group-Out CV.

    Args:
        X_tune: feature dataframe/array (n_samples, n_features)
        y_tune: dataframe/series/array; if DataFrame, should contain "CrossSection"
        g_tune: group labels (length n_samples), suitable for LeaveOneGroupOut
        param_space: dict of parameter distributions for RandomizedSearchCV
        base_kwargs: dict of fixed kwargs for XGBRegressor (e.g., objective, tree_method, etc.)

    Kwargs:
        n_iter: number of random parameter samples
        scoring: sklearn scoring string or callable
        random_state: RNG seed for reproducibility
        n_jobs: parallelism
        verbose: verbosity for RandomizedSearchCV
        refit: refit best model on full tuning set

    Returns:
        search: fitted RandomizedSearchCV object (use .best_estimator_, .best_params_, .cv_results_)
    """
    # ---- y handling (accept DataFrame with "CrossSection" or anything array-like)
    if hasattr(y_tune, "columns"):  # pandas DataFrame
        if "CrossSection" in y_tune.columns:
            y = y_tune["CrossSection"].to_numpy()
        elif y_tune.shape[1] == 1:
            y = y_tune.iloc[:, 0].to_numpy()
        else:
            raise ValueError('`y_tune` DataFrame must have a "CrossSection" column or exactly 1 column.')
    else:
        y = np.asarray(y_tune)

    y = np.ravel(y)
    groups = np.asarray(g_tune)

    if len(groups) != len(y):
        raise ValueError(f"Length mismatch: len(g_tune)={len(groups)} vs len(y_tune)={len(y)}")

    # ---- LOGO CV
    logo = LeaveOneGroupOut()

    # ---- model
    # Sensible defaults if user didn't provide them
    kwargs = dict(base_kwargs) if base_kwargs is not None else {}
    kwargs.setdefault("random_state", random_state)
    kwargs.setdefault("n_estimators", 500)
    kwargs.setdefault("verbosity", 0)

    model = XGBRegressor(**kwargs)

    # ---- random search
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_space,
        n_iter=int(n_iter),
        scoring=scoring,
        cv=logo,                 # groups are passed at fit-time
        refit=refit,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=True,
    )

    search.fit(X_tune, y, groups=groups)
    return search