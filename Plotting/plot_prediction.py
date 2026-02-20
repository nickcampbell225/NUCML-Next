import numpy as np
import matplotlib.pyplot as plt

# module-level storage
_X_HOLD = None
_Y_HOLD = None

def set_hold_data(x_hold, y_hold):
    """Call once to register holdout data for plotting."""
    global _X_HOLD, _Y_HOLD
    _X_HOLD = x_hold
    _Y_HOLD = y_hold

def plot_prediction(y_hold_pred, energy_col="Energy"):
    """
    Plot holdout truth vs prediction against _X_HOLD[energy_col].
    After calling set_hold_data(...), the only required input is y_hold_pred.
    """
    if _X_HOLD is None or _Y_HOLD is None:
        raise NameError("Holdout data not set. Call set_hold_data(x_hold, y_hold) first.")

    E = np.asarray(_X_HOLD[energy_col]).ravel()
    y_true = np.asarray(_Y_HOLD).ravel()
    y_pred = np.asarray(y_hold_pred).ravel()

    n = len(y_true)
    if len(E) != n or len(y_pred) != n:
        raise ValueError(f"Length mismatch: energy={len(E)}, y_hold={len(y_true)}, y_hold_pred={len(y_pred)}")

    order = np.argsort(E)
    E_s, yt, yp = E[order], y_true[order], y_pred[order]

    plt.figure()
    plt.plot(E_s, yt, label="True", linewidth=1.5)
    plt.plot(E_s, yp, label="Pred", linewidth=1.5)
    plt.xlabel("Energy")
    plt.ylabel("Target")
    plt.title("Holdout: True vs Pred vs Energy")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.scatter(E_s, (yp - yt), s=6, alpha=0.35)
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Energy")
    plt.ylabel("Residual (pred - true)")
    plt.title("Holdout residuals vs Energy")
    plt.tight_layout()
    plt.show()