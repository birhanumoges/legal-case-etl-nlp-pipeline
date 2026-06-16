"""time_series/visualization.py — Standalone time-series plots."""

from __future__ import annotations
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import PLOT_DIR
from utils.logger import get_logger

logger = get_logger(__name__)


def plot_annual_volume(annual: pd.Series, title: str = "Annual Case Volume"):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(annual.index, annual.values, alpha=0.3, color="steelblue")
    ax.plot(annual.index, annual.values, color="steelblue", linewidth=1.5)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("Cases")
    plt.tight_layout()
    path = PLOT_DIR / "ts_annual_volume.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Annual volume plot → %s", path)
    return path
