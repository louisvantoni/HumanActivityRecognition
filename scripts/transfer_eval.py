from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from config import LABEL_LIST, DATA_ROOT, DURATION
from data.transfer_dataset import TransferAudioDataset
from external.audioset_tagging_cnn.pytorch.models import Cnn14
from plotting import plot_multilabel_confusion
from util import generate_test_path_and_labels, save_report_and_metrics


def transfer_eval(feature_type="melspectrogram", epoch=1):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent
    checkpoint_dir = root_dir / "outputs/checkpoints_transfer"

    # === Dataset ===
    test_root = DATA_ROOT + "/Test_highpass"
    file_paths, labels = generate_test_path_and_labels(test_root, len(LABEL_LIST))

    dataset = TransferAudioDataset(file_paths=file_paths, labels=labels, sample_rate=32000, duration=DURATION)

    dataloader = DataLoader(
        dataset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True
    )

    # === Load model ===
    model = Cnn14(
        sample_rate=32000,
        window_size=1024,
        hop_size=320,
        mel_bins=64,
        fmin=50,
        fmax=14000,
        classes_num=527
    )

    # Replace final layer for 8-class task
    in_features = model.fc_audioset.in_features
    model.fc_audioset = nn.Linear(in_features, len(LABEL_LIST))

    # Load checkpoint
    checkpoint_path = checkpoint_dir / f"{feature_type}_pann_transfer_epoch_{epoch}.pth"
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    print(f"Loaded model from {checkpoint_path}")

    # === Evaluation ===
    criterion = nn.BCELoss()
    running_loss = 0.0
    all_outputs, all_targets = [], []

    with torch.no_grad():
        for waveforms, targets in dataloader:
            waveforms = waveforms.to(DEVICE)
            targets = targets.to(DEVICE).float()

            outputs = model(waveforms)
            probs = outputs["clipwise_output"]

            loss = criterion(probs, targets)
            running_loss += loss.item() * waveforms.size(0)

            all_outputs.append(probs.cpu())
            all_targets.append(targets.cpu())

    all_outputs = torch.cat(all_outputs).numpy()
    all_targets = torch.cat(all_targets).numpy()

    avg_loss = running_loss / len(dataset)
    print(f"\nTest BCE Loss: {avg_loss:.4f}")

    # === Predictions ===
    threshold = 0.5
    predicts = (all_outputs >= threshold).astype(int)

    #plot_multilabel_confusion(all_targets, predicts, LABEL_LIST)
    # Save metrics and report
    save_report_and_metrics(all_targets, all_outputs, predicts, avg_loss, feature_type, epoch, name="pann_transfer")


if __name__ == "__main__":
    for epoch in range(1,100):
        transfer_eval(feature_type="melspectrogram", epoch=epoch)