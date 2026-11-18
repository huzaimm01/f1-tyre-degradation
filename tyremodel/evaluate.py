"""Metrics and plots."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error

COLOURS = {"SOFT": "#d6402b", "MEDIUM": "#c98a00", "HARD": "#14202b"}


def mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))


def degradation_rates(df, column="corrected_s"):
    """Seconds lost per lap of tyre age, from a straight-line fit per compound."""
    rates = {}
    for compound, group in df.groupby("Compound"):
        if group["TyreLife"].nunique() >= 3:
            slope = np.polyfit(group["TyreLife"], group[column], 1)[0]
            rates[compound] = round(float(slope), 4)
    return rates


def plot_predicted_vs_actual(y_true, predictions, path):
    fig, axes = plt.subplots(1, len(predictions), figsize=(4 * len(predictions), 4), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    lo, hi = float(np.min(y_true)), float(np.max(y_true))
    for ax, (name, pred) in zip(axes, predictions.items()):
        ax.scatter(y_true, pred, s=12, alpha=0.6)
        ax.plot([lo, hi], [lo, hi], color="grey", linewidth=1)
        ax.set_title(f"{name} (MAE {mae(y_true, pred):.2f} s)")
        ax.set_xlabel("Actual lap time (s)")
    axes[0].set_ylabel("Predicted lap time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_degradation(df, path, column="corrected_s"):
    fig, ax = plt.subplots(figsize=(6, 4))
    for compound, group in df.groupby("Compound"):
        ax.scatter(group["TyreLife"], group[column], s=10, alpha=0.5,
                   color=COLOURS.get(compound), label=compound)
        if group["TyreLife"].nunique() >= 3:
            slope, intercept = np.polyfit(group["TyreLife"], group[column], 1)
            xs = np.array([group["TyreLife"].min(), group["TyreLife"].max()])
            ax.plot(xs, slope * xs + intercept, color=COLOURS.get(compound))
    ax.set_xlabel("Tyre age (laps)")
    ax.set_ylabel("Fuel-corrected lap time (s)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
