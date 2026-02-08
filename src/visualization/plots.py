"""
Visualization module for model analysis and results.
Generates publication-quality plots.
"""
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.base import BaseEstimator

logger = logging.getLogger("ml_tree_system")

# Set style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot confusion matrix as a heatmap.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        class_names: Names for each class.
        title: Plot title.
        save_path: Path to save the figure.
        figsize: Figure size.

    Returns:
        Matplotlib figure object.
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names or np.unique(y_true),
        yticklabels=class_names or np.unique(y_true),
        ax=ax
    )

    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved confusion matrix to: {save_path}")

    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 15,
    title: str = "Feature Importance",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot feature importance as a horizontal bar chart.

    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns.
        top_n: Number of top features to show.
        title: Plot title.
        save_path: Path to save the figure.
        figsize: Figure size.

    Returns:
        Matplotlib figure object.
    """
    # Get top N features
    plot_data = importance_df.head(top_n).copy()
    plot_data = plot_data.sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(plot_data)))

    bars = ax.barh(
        plot_data["feature"],
        plot_data["importance"],
        color=colors
    )

    # Add value labels
    for bar, val in zip(bars, plot_data["importance"]):
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            fontsize=9
        )

    ax.set_xlabel("Importance Score", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlim(0, plot_data["importance"].max() * 1.15)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved feature importance plot to: {save_path}")

    return fig


def plot_decision_tree(
    model: DecisionTreeClassifier,
    feature_names: List[str],
    class_names: List[str],
    max_depth: int = 3,
    title: str = "Decision Tree Visualization",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (20, 10)
) -> plt.Figure:
    """
    Visualize decision tree structure.

    Args:
        model: Trained DecisionTreeClassifier.
        feature_names: List of feature names.
        class_names: List of class names.
        max_depth: Maximum depth to display.
        title: Plot title.
        save_path: Path to save the figure.
        figsize: Figure size.

    Returns:
        Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        max_depth=max_depth,
        fontsize=10,
        ax=ax
    )

    ax.set_title(title, fontsize=16, fontweight="bold")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved decision tree plot to: {save_path}")

    return fig


def export_tree_graphviz(
    model: DecisionTreeClassifier,
    feature_names: List[str],
    class_names: List[str],
    save_path: str
) -> str:
    """
    Export decision tree to GraphViz DOT format.

    Args:
        model: Trained DecisionTreeClassifier.
        feature_names: List of feature names.
        class_names: List of class names.
        save_path: Path to save the DOT file.

    Returns:
        DOT string representation.
    """
    dot_data = export_graphviz(
        model,
        out_file=None,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        special_characters=True
    )

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        f.write(dot_data)

    logger.info(f"Exported tree to GraphViz: {save_path}")
    return dot_data


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot ROC curve for binary classification.

    Args:
        y_true: True labels.
        y_prob: Prediction probabilities for positive class.
        model_name: Name of the model for legend.
        save_path: Path to save the figure.
        figsize: Figure size.

    Returns:
        Matplotlib figure object.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        fpr, tpr,
        color="darkorange",
        lw=2,
        label=f"{model_name} (AUC = {roc_auc:.3f})"
    )
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Receiver Operating Characteristic (ROC)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved ROC curve to: {save_path}")

    return fig


def plot_roc_comparison(
    models_data: dict,
    y_true: np.ndarray,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot ROC curves for multiple models on same figure.

    Args:
        models_data: Dict of {model_name: y_prob}.
        y_true: True labels.
        save_path: Path to save figure.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(models_data)))

    for (name, y_prob), color in zip(models_data.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--", label="Random")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved ROC comparison to: {save_path}")

    return fig


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    title: str = "Model Comparison",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot bar chart comparing models across metrics.

    Args:
        comparison_df: DataFrame with model comparison results.
        metrics: List of metrics to include.
        title: Plot title.
        save_path: Path to save figure.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    if metrics is None:
        # Exclude 'Model' column
        metrics = [c for c in comparison_df.columns if c != "Model"]

    plot_data = comparison_df.set_index("Model")[metrics]

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(metrics))
    width = 0.35
    models = plot_data.index.tolist()

    for i, model in enumerate(models):
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, plot_data.loc[model], width, label=model)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9
            )

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved model comparison to: {save_path}")

    return fig
