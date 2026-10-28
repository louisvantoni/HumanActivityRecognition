import json
import random
import re
from pathlib import Path

import matplotlib
import pandas as pd

import librosa.feature, librosa.display
import os
import librosa
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
import torch
from matplotlib import pyplot as plt, ticker
from pydub import AudioSegment
from scipy import signal
from scipy.signal import butter, lfilter
from sklearn.metrics import roc_curve, auc, f1_score, precision_score, recall_score, accuracy_score, \
    average_precision_score, classification_report, label_ranking_average_precision_score
from torchvision.ops import DropBlock2d

from config import SAMPLE_RATE, LABEL_LIST, METRICS_PATH, DATA_ROOT

AUDIO_EXTENSIONS = (".mp3", ".wav", ".flac", ".aac", ".m4a")


class ShiftSpectrogramX:
    """
    Cyclically shifts a spectrogram along the time axis (x-axis) and wraps around.
    shift can be:
      - an integer (fixed shift)
      - a tuple (min_shift, max_shift) → random shift in this range
    """
    def __init__(self, shift):
        self.shift = shift

    def __call__(self, spec: torch.Tensor):
        if isinstance(self.shift, tuple):
            k = random.randint(*self.shift)
        else:
            k = self.shift

        return torch.roll(spec, shifts=k, dims=-1)

def resample_audio(input_folder, output_folder, target_rate=44100):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.wav'):
            filepath = os.path.join(input_folder, filename)
            print(f"Processing {filename}...")

            # Load audio
            audio = AudioSegment.from_file(filepath)

            # Resample
            audio = audio.set_frame_rate(target_rate)

            filename = filename.split(".")[0] + "-8" + ".wav"
            # Save with same name in output folder
            output_path = os.path.join(output_folder, filename)
            audio.export(output_path, format=filename.split('.')[-1])
            print(f"Saved to {output_path}")

def highpass_filter(data, sr=SAMPLE_RATE, cutoff=50, order=4):
    """Apply a Butterworth high-pass filter."""
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    # Frequency response (digital)
    w, h = signal.freqz(b, a, worN=2048)

    # Convert rad/sample → Hz
    freqs = w * sr / (2 * np.pi)

    filtered_data = lfilter(b, a, data)
    return filtered_data

def normalize_loudness_audio_file(input_path, output_path, target_lufs):
    try:
        y, sr = librosa.load(input_path, sr=None, mono=True)
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(y)
        print(f"Original LUFS: {loudness:.2f} for {input_path}")
        y_norm = pyln.normalize.loudness(y, loudness, target_lufs)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, y_norm, sr)
        print(f"Saved normalized file: {output_path}\n")
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

def split_audio_into_chunks(folder_path, output_folder_path, chunk_length_in_seconds=6):
    """
    Split all audio files in a folder into fixed-length chunks and save
    them as WAV files in another folder. The final chunk of each file is
    padded with silence so every chunk has exactly the same length.

    Args:
        folder_path (str): Folder containing the audio files.
        output_folder_path (str): Folder where the chunks are saved.
        chunk_length_in_seconds (int | float): Length of each chunk. Default 6.
    """
    os.makedirs(output_folder_path, exist_ok=True)

    chunk_ms = int(chunk_length_in_seconds * 1000)

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(AUDIO_EXTENSIONS):
            continue

        audio_file_path = os.path.join(folder_path, filename)
        audio = AudioSegment.from_file(audio_file_path)
        base_name = os.path.splitext(filename)[0]

        chunk_count = 0
        for start in range(0, len(audio), chunk_ms):
            chunk = audio[start:start + chunk_ms]

            # Pad the last chunk with silence if it is too short
            if len(chunk) < chunk_ms:
                silence = AudioSegment.silent(
                    duration=chunk_ms - len(chunk),
                    frame_rate=audio.frame_rate,
                )
                chunk = chunk + silence

            output_filename = f"{base_name}_chunk_{chunk_count:03d}.wav"
            output_file_path = os.path.join(output_folder_path, output_filename)
            chunk.export(output_file_path, format="wav")
            chunk_count += 1

        print(f"Split {filename} into {chunk_count} chunk(s)")

def cut_or_pad_audio_to_length(folder_path, output_folder_path, length_in_seconds):
    """
    Cut all audio files in a folder to a given length, or pad them with
    silence if they are shorter. Results are saved as WAV files.

    Args:
        folder_path (str): Folder containing the audio files.
        output_folder_path (str): Folder where the processed files are saved.
        length_in_seconds (int | float): Target length in seconds.
    """
    os.makedirs(output_folder_path, exist_ok=True)

    target_ms = int(length_in_seconds * 1000)

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(AUDIO_EXTENSIONS):
            continue

        audio_file_path = os.path.join(folder_path, filename)
        audio = AudioSegment.from_file(audio_file_path)

        if len(audio) > target_ms:
            # Cut to the target length
            processed = audio[:target_ms]
        elif len(audio) < target_ms:
            # Pad with silence matching the original audio's format
            silence = AudioSegment.silent(
                duration=target_ms - len(audio),
                frame_rate=audio.frame_rate,
            )
            processed = audio + silence
        else:
            processed = audio

        output_filename = os.path.splitext(filename)[0] + ".wav"
        output_file_path = os.path.join(output_folder_path, output_filename)
        processed.export(output_file_path, format="wav")

        print(f"Processed {filename} to {length_in_seconds} seconds "
              f"and saved to {output_file_path}")

def generate_train_paths_and_labels(data_root):
    file_paths = []
    labels = []

    class_names = sorted([
        d for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
    ])

    print(f"Found classes: {class_names}")

    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}

    for cls_name in class_names:
        cls_folder = os.path.join(data_root, cls_name)

        for root, _, files in os.walk(cls_folder):
            for file in files:
                if file.endswith(".wav"):

                    path = os.path.join(root, file)
                    file_paths.append(path)

                    label = [0] * len(class_names)
                    label[class_to_idx[cls_name]] = 1
                    labels.append(label)

    return file_paths, labels

def generate_test_path_and_labels(data_root, num_classes=8):
    file_paths = []
    labels = []

    for root, _, files in os.walk(data_root):
        for file in files:
            if file.endswith(".wav"):
                path = os.path.join(root, file)
                file_paths.append(path)

                match = re.search(r'(\d+)(\.[^.]+)?$', file)
                if match:
                    digits = match.group(1)
                    label_indices = [(int(ch) - 1) for ch in digits]
                else:
                    raise ValueError(f"Could not extract label digits from filename: {file}")

                # Convert to multi-hot
                multi_hot = torch.zeros(num_classes, dtype=torch.float)
                for idx in label_indices:
                    if idx >= num_classes:
                        raise ValueError(f"Label index {idx} in {file} exceeds num_classes={num_classes}")
                    elif idx < 0:
                        continue
                    multi_hot[idx] = 1

                labels.append(multi_hot)


    return file_paths, labels

def extract_melspectrogram(y, sr=SAMPLE_RATE, n_mels=128, n_fft=2048, hop_length=512):

    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        power=1,
        window="hamming"
    )
    S_pcen = librosa.pcen(S * (2 ** 31), sr=SAMPLE_RATE,hop_length=hop_length, time_constant=0.4)

    S_pcen[:, :10] = 0.0


    S_dB = librosa.power_to_db(S, ref=np.max)

    return S_pcen

def extract_mfcc(y, sr=SAMPLE_RATE, n_fft=2048,hop_length=512, n_mfcc=13):
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mfcc=n_mfcc)
    return mfcc

def dropblock_scheduler(epoch, total_epochs, start_prob=0.0, end_prob=0.2):
    return start_prob + (end_prob - start_prob) * (epoch / total_epochs)

def set_drop_prob(model, drop_prob):
    for module in model.modules():
        if isinstance(module, DropBlock2d):
            module.drop_prob = drop_prob

def summix_batch(spectrograms, labels, normalize=True):
    """
    Fully vectorized additive mixing augmentation for spectrograms, avoiding
    mixing spectrograms with overlapping labels (multi-hot). If no valid pairing exists,
    the sample is left unchanged.

    Args:
        spectrograms (Tensor): Batch of spectrograms (B, C, H, W)
        labels (Tensor): Batch of multi-hot labels (B, num_classes)
        normalize (bool): Whether to normalize each mixed spectrogram

    Returns:
        mixed_spectrograms (Tensor): same shape as input
        mixed_labels (Tensor): same shape as input
    """
    B, C, H, W = spectrograms.shape
    device = spectrograms.device

    # Compute overlap matrix (B, B)
    overlap = labels @ labels.T  # counts of overlapping classes
    valid_mask = (overlap == 0)
    valid_mask.fill_diagonal_(False)  # prevent self-pairing

    # For each row, get indices of valid pairs
    valid_indices = [valid_mask[i].nonzero(as_tuple=False).flatten() for i in range(B)]

    # Randomly pick one valid pair for each sample
    # If no valid pair exists, pick self (no mixing)
    paired_indices = torch.arange(B, device=device)
    for i, inds in enumerate(valid_indices):
        if len(inds) > 0:
            paired_indices[i] = inds[torch.randint(len(inds), (1,), device=device)]

    # Additive mixing
    mixed_spectrograms = spectrograms + spectrograms[paired_indices]

    # Optional normalization
    if normalize:
        mixed_spectrograms = mixed_spectrograms / (mixed_spectrograms.amax(dim=(2,3), keepdim=True) + 1e-8)

    # Union of labels
    mixed_labels = torch.clamp(labels + labels[paired_indices], 0, 1)

    return mixed_spectrograms, mixed_labels

def show_spec(spec, sr=44100, hop_length=512, y_axis='mel', fmax=None):
    """
    Display a single spectrogram using librosa.display.specshow.

    Parameters:
    - spec: torch.Tensor or np.ndarray, shape (1, F, T) or (F, T)
    - sr: sample rate
    - title: plot title
    - hop_length: hop length used to compute the spectrogram
    - y_axis: 'mel', 'hz', or None
    - fmax: maximum frequency for display (Hz)
    """
    import numpy as np

    # Convert to numpy if torch tensor
    if isinstance(spec, torch.Tensor):
        spec = spec.detach().cpu().numpy()

    # Remove channel dim if present
    if spec.ndim == 3 and spec.shape[0] == 1:
        spec = spec[0]

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman"],
        "axes.unicode_minus": False,
    })
    plt.figure(figsize=(10, 4), dpi=300, constrained_layout=True)
    librosa.display.specshow(spec, sr=sr, hop_length=hop_length, x_axis='time', y_axis=y_axis, fmax=fmax)
    plt.show()


def classify_with_thresholds(probabilities, thresholds):
    probabilities = np.array(probabilities)
    thresholds = np.array(thresholds)

    # Apply thresholds
    predictions = (probabilities > thresholds).astype(int)

    # Ensure at least one 1
    if predictions.sum() == 0:
        max_index = np.argmax(probabilities)
        predictions[max_index] = 1

    return predictions

def get_best_epoch_f1(feature_type, name):
    epoch_pattern = re.compile(fr"{feature_type}_{name}_cnn_eval_report_epoch_(\d+).json")

    best_scores = {}

    for filename in os.listdir(METRICS_PATH):
        match = epoch_pattern.match(filename)
        if not match:
            continue

        epoch = int(match.group(1))
        filepath = os.path.join(METRICS_PATH, filename)

        with open(filepath, 'r') as f:
            data = json.load(f)

        for class_name, metrics in data.items():
            if not isinstance(metrics, dict):
                continue
            if class_name in ["micro avg", "macro avg", "weighted avg", "samples avg"]:
                continue

            f1 = metrics.get("f1-score", 0.0)
            if class_name not in best_scores or f1 > best_scores[class_name][1]:
                best_scores[class_name] = (epoch, f1)

    print("Best epochs for each class based on f1-score:")
    for class_name, (epoch, f1) in best_scores.items():
        print(f"{class_name}: Epoch {epoch} with f1-score {f1:.4f}")

def get_best_epoch_score(feature_type, name, scores_to_extract):
    """
    Extracts the best epochs based on given scores (e.g., 'mean_average_precision')
    from JSON evaluation reports.

    Args:
        feature_type (str): Feature type prefix in filename.
        name (str): Model or dataset name part in filename.
        scores_to_extract (list[str]): List of score keys to extract.

    Returns:
        dict: {score_name: {"best_epoch": int, "best_value": float}}
    """
    epoch_pattern = re.compile(fr"{feature_type}_{name}_cnn_eval_report_epoch_(\d+)\.json")
    best_scores = {score: {"best_epoch": None, "best_value": float('-inf')} for score in scores_to_extract}

    for filename in os.listdir(METRICS_PATH):
        match = epoch_pattern.match(filename)
        if not match:
            continue

        epoch = int(match.group(1))
        filepath = os.path.join(METRICS_PATH, filename)

        with open(filepath, 'r') as f:
            data = json.load(f)

        for score in scores_to_extract:
            if score in data:
                value = data[score]
                if value > best_scores[score]["best_value"]:
                    best_scores[score] = {"best_epoch": epoch, "best_value": value}

    print(best_scores)


def get_best_epoch_auc(feature_type, name):
    epoch_pattern = re.compile(fr"{feature_type}_{name}_cnn_eval_report_epoch_(\d+).json")

    best_scores = {}

    for filename in os.listdir(METRICS_PATH):
        match = epoch_pattern.match(filename)
        if not match:
            continue

        epoch = int(match.group(1))
        filepath = os.path.join(METRICS_PATH, filename)

        with open(filepath, 'r') as f:
            data = json.load(f)

        for class_name, metrics in data.items():
            if not isinstance(metrics, dict):
                continue
            if class_name in ["micro avg", "macro avg", "weighted avg", "samples avg"]:
                continue

            roc_auc = metrics.get("roc_auc", 0.0)
            if class_name not in best_scores or roc_auc > best_scores[class_name][1]:
                best_scores[class_name] = (epoch, roc_auc)

    print("Best epochs for each class based on auc:")
    for class_name, (epoch, roc_auc) in best_scores.items():
        print(f"{class_name}: Epoch {epoch} with auc {roc_auc:.4f}")

def save_report_and_metrics(targets, outputs, predicts, loss, feature_type, epoch, name):
    f1 = f1_score(targets, predicts, average='samples')
    precision = precision_score(targets, predicts, average='samples')
    recall = recall_score(targets, predicts, average='samples')
    subset_acc = accuracy_score(targets, predicts)

    mAP = average_precision_score(targets, outputs, average="macro")
    labelAP = label_ranking_average_precision_score(targets, outputs)

    num_classes = targets.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(targets[:, i], outputs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot ROC curve for this class
        matplotlib.use('Agg')
        plt.figure()
        plt.plot(fpr[i], tpr[i], label=f'ROC Curve (AUC = {roc_auc[i]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=1)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - Class {i}')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()

        # Save the figure
        label_name = LABEL_LIST[i]
        figure_dir = Path(METRICS_PATH) / "figures"
        figure_dir.mkdir(parents=True, exist_ok=True)

        plt.savefig(figure_dir / f"{feature_type}_{name}_roc_{label_name}_epoch_{epoch}.png")
        plt.close()

    report_dict = classification_report(
        targets,
        predicts,
        target_names=LABEL_LIST,
        output_dict=True
    )

    report_dict["BCE_validation_loss"] = loss
    report_dict["subset_accuracy"] = subset_acc
    report_dict["mean_average_precision"] = mAP
    report_dict["label_ranking_average_precision"] = labelAP
    for i, class_name in enumerate(LABEL_LIST):
        if class_name in report_dict and i in roc_auc:
            report_dict[class_name]["roc_auc"] = roc_auc[i]

    report_str = classification_report(
        targets,
        predicts,
        target_names=LABEL_LIST,
        output_dict=False
    )

    full_report_str = (
            f"Mean Average Precision (mAP): {mAP:.4f}\n"
            f"Label Ranking Average Precision (LRAP): {labelAP:.4f}\n"
            f"Subset Accuracy (exact match): {subset_acc:.4f}\n"
            f"Multi-label F1 (samples avg): {f1:.4f}\n"
            f"Multi-label Precision (samples avg): {precision:.4f}\n"
            f"Multi-label Recall (samples avg): {recall:.4f}\n\n"
            "=== Classification Report (per label) ===\n"
            + report_str
    )

    print("\nFinal Evaluation Report:\n")
    print(full_report_str)

    metrics_dir = Path(METRICS_PATH)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    metrics_txt_path = metrics_dir / f"{feature_type}_{name}_cnn_eval_report_epoch_{epoch}.txt"
    with open(metrics_txt_path, "w") as f:
        f.write(full_report_str)

    metrics_json_path = metrics_dir / f"{feature_type}_{name}_cnn_eval_report_epoch_{epoch}.json"
    with open(metrics_json_path, "w") as f:
        json.dump(report_dict, f, indent=4)

    print(f"\nReports saved to: {metrics_dir}/")

def plot_bce_loss_comparison(
    eval_folder,
    checkpoint_folder,
    model_name="mono",
    metric_name="mean_average_precision"
):
    """
    Plots comparison between training losses and evaluation metric (e.g., mAP) over epochs.

    Args:
        eval_folder (str): Path to folder with *eval_report_epoch_X.json files.
        checkpoint_folder (str): Path to folder with model checkpoint files containing training losses.
        model_name (str): Model name identifier within filenames (e.g., "mono" or "cnn").
        metric_name (str): Metric name key in evaluation JSONs (default: "mean_average_precision").
    """

    # --- Regex patterns ---
    eval_pattern = re.compile(rf"melspectrogram_{model_name}_cnn_eval_report_epoch_(\d+)\.json")
    train_pattern = re.compile(r"melspectrogram_cnn_model_epoch_(\d+)\.pth")

    maps = {}
    train_losses = {}

    # --- Parse evaluation metrics ---
    for filename in os.listdir(eval_folder):
        match = eval_pattern.match(filename)
        if match:
            epoch = int(match.group(1))
            filepath = os.path.join(eval_folder, filename)
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                if metric_name in data:
                    maps[epoch] = data[metric_name]
            except Exception as e:
                print(f"Could not read {filename}: {e}")

    # --- Parse training losses ---
    for filename in os.listdir(checkpoint_folder):
        match = train_pattern.match(filename)
        if match:
            epoch = int(match.group(1))
            filepath = os.path.join(checkpoint_folder, filename)
            try:
                checkpoint = torch.load(filepath, map_location="cpu")
                if "loss" in checkpoint:
                    train_losses[epoch] = checkpoint["loss"]
                else:
                    print(f"No 'loss' key in {filename}, skipping.")
            except Exception as e:
                print(f"Could not read {filename}: {e}")

    # --- Prepare data ---
    has_maps = bool(maps)
    has_losses = bool(train_losses)
    if not has_maps and not has_losses:
        print("No valid data found to plot.")
        return

    # --- Plot setup ---
    plt.style.use("seaborn-v0_8-dark-palette")
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman"],
        "axes.unicode_minus": False,
    })
    fig_width_pt = 345
    inches_per_pt = 1.0 / 72.27
    fig_width_in = fig_width_pt * inches_per_pt
    fig, ax = plt.subplots(figsize=(fig_width_in, fig_width_in*0.6), dpi=300, constrained_layout=True)

    if has_maps:
        epochs = sorted(maps.keys())
        values = [maps[e] for e in epochs]
        ax.plot(
            epochs,
            values,
            marker=',',
            linewidth=2.2,
            label={metric_name.title().replace("_", " ")}
        )

    if has_losses:
        epochs_t = sorted(train_losses.keys())
        values_t = [train_losses[e] for e in epochs_t]
        ax.plot(
            epochs_t,
            values_t,
            marker=',',
            linewidth=2.2,
            label="Training Loss"
        )

    # --- Axis formatting ---
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(f"BCE Loss / mAP", fontsize=12)

    # Start axes at 0
    ax.set_xlim(left=0, right=100)
    ax.set_ylim(bottom=0)

    # Add subticks
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(which='major', length=6, width=1)
    ax.tick_params(which='minor', length=3, width=0.8, color='gray')
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    plt.show()


def compute_class_metric_correlation(folder, model_name, metric):
    """
    Extracts the given metric per class across epochs
    and computes correlation between classes.

    Args:
        folder (str): Path to folder containing eval JSON reports.
        model_name (str): Model identifier (e.g., "mono")
        metric (str): Metric to extract (e.g., "precision", "roc_auc").

    Returns:
        pandas.DataFrame: Correlation matrix of metric across classes.
    """

    pattern = re.compile(rf"melspectrogram_{model_name}_cnn_eval_report_epoch_(\d+)\.json")

    # { class_name: {epoch: metric_value} }
    data = {}

    for filename in os.listdir(folder):
        match = pattern.match(filename)
        if not match:
            continue
        epoch = int(match.group(1))
        filepath = os.path.join(folder, filename)

        with open(filepath, 'r') as f:
            report = json.load(f)

        for class_name, values in report.items():
            if isinstance(values, dict) and metric in values:
                if class_name not in data:
                    data[class_name] = {}
                data[class_name][epoch] = values[metric]

    # Convert to DataFrame (rows = epochs, columns = classes)
    df = pd.DataFrame(data).sort_index()

    # Compute correlation matrix
    corr_matrix = df.corr()

    return corr_matrix

if __name__ == "__main__":
