import matplotlib.pyplot as plt
import numpy as np

from config import LABEL_LIST, DATA_ROOT
from util import generate_test_path_and_labels


def plot_multi_hot_label_counts(multi_hot_labels, label_names=None):
    """
    Plots the occurrence counts of each label from multi-hot encoded vectors.

    Parameters:
        multi_hot_labels (list of list[int]): Multi-hot encoded labels.
        label_names (list[str], optional): Names for each label index.
    """
    # Convert to numpy array
    arr = np.array(multi_hot_labels)

    # Sum occurrences for each label column
    label_counts = arr.sum(axis=0)

    # Default label names if not provided
    if label_names is None:
        label_names = [f"Label {i}" for i in range(arr.shape[1])]

    # Plot
    plt.style.use("seaborn-v0_8-dark-palette")
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman"],
        "axes.unicode_minus": False,
    })
    plt.figure(figsize=(10, 6), dpi=300, constrained_layout=True)
    plt.bar(label_names, label_counts)
    plt.xlabel("Labels")
    plt.ylabel("Occurrences")
    plt.xticks(rotation=45)
    plt.show()

if __name__ == "__main__":
    path = DATA_ROOT + "/Train_highpass"
    paths, labels = generate_test_path_and_labels(path)
    label_names = LABEL_LIST
    plot_multi_hot_label_counts(labels, label_names)