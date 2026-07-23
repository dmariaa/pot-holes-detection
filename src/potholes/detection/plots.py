import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix

from potholes.detection.data.dataset import RoadLabel


def confusion_matrix_values(labels_np: np.ndarray, preds_np: np.ndarray) -> tuple[list[str], np.ndarray, np.ndarray]:
    class_ids = [label.value for label in RoadLabel]
    class_names = [label.label for label in RoadLabel]
    cm = confusion_matrix(labels_np, preds_np, labels=class_ids)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm.astype(float),
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0,
    )
    return class_names, cm, cm_norm


def confusion_matrix_figure(
        labels_np: np.ndarray,
        preds_np: np.ndarray,
        *,
        title: str = "Normalized Confusion Matrix",
) -> go.Figure:
    class_names, cm, cm_norm = confusion_matrix_values(labels_np, preds_np)

    fig = go.Figure(
        data=go.Heatmap(
            z=cm_norm,
            x=class_names,
            y=class_names,
            zmin=0,
            zmax=1,
            colorscale="Blues",
            customdata=cm,
            text=np.round(cm_norm, 2),
            texttemplate="%{text}",
            hovertemplate=(
                "True: %{y}<br>"
                "Pred: %{x}<br>"
                "Normalized: %{z:.3f}<br>"
                "Count: %{customdata}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
    )
    return fig


def write_confusion_matrix_image(
        labels_np: np.ndarray,
        preds_np: np.ndarray,
        output_path: str,
        *,
        title: str = "Normalized Confusion Matrix",
) -> go.Figure:
    fig = confusion_matrix_figure(labels_np, preds_np, title=title)
    fig.write_image(output_path)
    return fig
