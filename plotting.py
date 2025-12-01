import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from sklearn.metrics import multilabel_confusion_matrix


def plot_multilabel_confusion(y_true, y_pred, class_names=None):
    """
    Compute and plot multilabel confusion matrices using scikit-learn.

    Parameters:
    y_true : np.ndarray (n_samples, n_classes)
        Ground truth binary indicators.
    y_pred : np.ndarray (n_samples, n_classes)
        Predicted binary indicators.
    class_names : list of str
        Names of the classes.
    """
    conf_matrices = multilabel_confusion_matrix(y_true, y_pred)
    n_classes = conf_matrices.shape[0]

    cols = int(np.ceil(np.sqrt(n_classes)))
    rows = int(np.ceil(n_classes / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()

    for i, cm in enumerate(conf_matrices):
        ax = axes[i]
        im = ax.imshow(cm, interpolation='nearest')
        ax.set_title(f"Class {class_names[i] if class_names else i}")

        # Add text annotations
        thresh = cm.max() / 2.
        for x, y in product(range(2), range(2)):
            ax.text(y, x, cm[x, y],
                    ha="center", va="center",
                    color="black")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"])
        ax.set_yticklabels(["True 0", "True 1"])

    # Remove unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(pad=3.0)
    plt.show()
