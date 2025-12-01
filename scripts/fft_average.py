import os
from pathlib import Path

import numpy as np
import librosa
import matplotlib.pyplot as plt

from config import DATA_ROOT, LABEL_LIST

# ------------------------------
# Configuration
# ------------------------------
train_folder = DATA_ROOT + "/Train_highpass"
test_folder = DATA_ROOT + "/Test_highpass"

num_classes = 8
sample_rate = 44100

# ------------------------------
# Helper: process folder to compute avg FFT per class
# ------------------------------
def compute_fft_averages(folder):
    fft_sums = {cls: None for cls in range(num_classes)}
    fft_counts = {cls: 0 for cls in range(num_classes)}

    for file in os.listdir(folder):
        if not file.lower().endswith(".wav"):
            continue

        name_without_ext = os.path.splitext(file)[0]

        # Extract labels after the last "-"
        if "-" not in name_without_ext:
            print(f"Skipping (no '-' found): {file}")
            continue

        label_part = name_without_ext.split("-")[-1]

        # Convert each character into a class label
        try:
            labels = [(int(ch) - 1) for ch in label_part]
        except ValueError:
            print(f"Skipping (invalid label chars): {file}")
            continue

        audio_path = os.path.join(folder, file)
        y, sr = librosa.load(audio_path, sr=sample_rate)

        # FFT
        fft_vals = np.abs(np.fft.rfft(y))
        fft_vals = fft_vals / np.max(fft_vals)  # normalize

        for cls in labels:  # multi-class labeling
            if cls not in fft_sums:
                continue

            if fft_sums[cls] is None:
                fft_sums[cls] = fft_vals.copy()
            else:
                # PAD TO SAME LENGTH BEFORE ADDING
                max_len = max(len(fft_sums[cls]), len(fft_vals))
                fft_sums[cls] = np.pad(fft_sums[cls], (0, max_len - len(fft_sums[cls])), mode='constant')
                fft_vals = np.pad(fft_vals, (0, max_len - len(fft_vals)), mode='constant')
                fft_sums[cls] += fft_vals

            fft_counts[cls] += 1


    # Compute average FFT for each class
    fft_avg = {
        cls: fft_sums[cls] / fft_counts[cls]
        for cls in fft_sums
        if fft_counts[cls] > 0
    }

    return fft_avg

# ------------------------------
# Compute averages
# ------------------------------
train_fft = compute_fft_averages(train_folder)
test_fft = compute_fft_averages(test_folder)

# Frequency axis for plotting
max_fft_len = max(max(len(v) for v in train_fft.values()), max(len(v) for v in test_fft.values()))
freqs = np.fft.rfftfreq((max_fft_len - 1) * 2 , 1 / sample_rate)

# ------------------------------
# Plotting
# ------------------------------
plt.style.use("seaborn-v0_8-dark-palette")
plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman"],
        "axes.unicode_minus": False,
})
script_dir = Path(__file__).resolve().parent
root_dir = script_dir.parent
checkpoint_dir = root_dir / "outputs"
output_dir = checkpoint_dir / "fft_plots"
os.makedirs(output_dir, exist_ok=True)

all_values = []

for cls in (set(train_fft.keys()) | set(test_fft.keys())):
    if cls in train_fft:
        all_values.append(np.max(train_fft[cls]))
    if cls in test_fft:
        all_values.append(np.max(test_fft[cls]))

global_max = max(all_values)
print(f"Global max amplitude = {global_max}")

for cls in sorted(train_fft.keys() | test_fft.keys()):

    plt.figure(figsize=(10, 6), dpi=300, constrained_layout=True)

    if cls in train_fft:
        plt.plot(freqs[:len(train_fft[cls])], train_fft[cls], label="Train", alpha=0.7)

    if cls in test_fft:
        plt.plot(freqs[:len(test_fft[cls])], test_fft[cls], label="Test", alpha=0.7)

    plt.xlabel("Frequency (Hz)", fontsize=24)
    plt.ylabel("Amplitude (normalized)", fontsize=24)
    plt.xscale("log")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.ylim([0, global_max * 1.1])
    plt.xlim(left=10)

    filename = os.path.join(output_dir, f"fft_class_{LABEL_LIST[cls]}.png")
    plt.savefig(filename, dpi=300)
    plt.close()

    print(f"Saved: {filename}")