"""visualization/shap_plots.py — SHAP feature importance bar chart."""

from __future__ import annotations
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from config import PLOT_DIR, REPORT_DIR
from utils.logger import get_logger

logger = get_logger(__name__)


def plot_shap_summary(target_name: str, top_n: int = 20):
    """Read SHAP JSON report and plot bar chart of top features."""
    json_path = REPORT_DIR / f"{target_name}_shap_summary.json"
    if not json_path.exists():
        logger.warning("No SHAP report for %s", target_name)
        return

    with open(json_path) as f:
        data = json.load(f)

    feats  = data.get("top_features", [])[:top_n]
    names  = [d["feature"] for d in feats]
    values = [d["mean_abs_shap"] for d in feats]

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.35)))
    ax.barh(names[::-1], values[::-1], color="steelblue")
    ax.set_title(f"SHAP Feature Importance — {target_name}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Mean |SHAP value|")
    plt.tight_layout()

    path = PLOT_DIR / f"shap_{target_name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("SHAP plot saved → %s", path)
    return path
