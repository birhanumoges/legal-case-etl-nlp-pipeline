"""visualization/rag_visualization.py — RAG evaluation charts."""

from __future__ import annotations
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from config import PLOT_DIR, REPORT_DIR
from utils.logger import get_logger

logger = get_logger(__name__)


def plot_rag_evaluation():
    path = REPORT_DIR / "rag_evaluation.json"
    if not path.exists():
        return
    with open(path) as f:
        data = json.load(f)

    summary = data.get("summary", {})
    metrics = {k: v for k, v in summary.items() if isinstance(v, float)}
    if not metrics:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(list(metrics.keys()), list(metrics.values()), color="teal")
    ax.set_xlim(0, 1)
    ax.set_title("RAG Evaluation Metrics", fontsize=12, fontweight="bold")
    ax.set_xlabel("Score (0–1)")
    plt.tight_layout()

    out = PLOT_DIR / "rag_evaluation.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("RAG evaluation plot → %s", out)
    return out
