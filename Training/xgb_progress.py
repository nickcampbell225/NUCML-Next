# xgb_progress.py
import xgboost as xgb
from tqdm.auto import tqdm

class TQDMProgress(xgb.callback.TrainingCallback):
    """
    Reusable tqdm progress bar for XGBoost sklearn API.
    If total is None, tries to infer from model.get_params()["n_estimators"].
    """
    def __init__(self, total=None, desc="xgboost", unit="tree"):
        self.total = total
        self.desc = desc
        self.unit = unit
        self.pbar = None

    def _infer_total(self, model):
        if self.total is not None:
            return int(self.total)
        if hasattr(model, "get_params"):
            n = model.get_params().get("n_estimators")
            if n is not None:
                return int(n)
        return None  # tqdm will still work, but without %/ETA

    def before_training(self, model):
        self.pbar = tqdm(total=self._infer_total(model), desc=self.desc, unit=self.unit)
        return model

    def after_iteration(self, model, epoch, evals_log):
        self.pbar.update(1)
        return False

    def after_training(self, model):
        if self.pbar is not None:
            self.pbar.close()
        return model


def attach_xgb_progress(model, desc="xgboost", total=None, unit="tree"):
    """
    Returns the same model with the progress callback attached (xgboost 3.2.0 style:
    callbacks are passed as a model parameter, not fit()).
    """
    model.set_params(callbacks=[TQDMProgress(total=total, desc=desc, unit=unit)])
    return model