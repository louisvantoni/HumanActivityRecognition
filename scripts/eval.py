from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch import nn

from models.cnn import AudioCNN
from data.dataset import AudioDataset
from config import SAMPLE_RATE, LABEL_LIST, DATA_ROOT

from util import generate_test_path_and_labels, save_report_and_metrics, classify_with_thresholds


def evaluation(feature_type="melspectrogram", epoch=1):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent
    checkpoint_dir = root_dir / "outputs/checkpoints"

    test_root = DATA_ROOT + "/Test_highpass"
    file_paths, labels = generate_test_path_and_labels(test_root, 8)

    dataset = AudioDataset(
        file_paths=file_paths,
        labels=labels,
        sample_rate=SAMPLE_RATE,
        feature_type=feature_type
    )

    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)

    #  Load trained model
    model = AudioCNN(num_classes=8)

    checkpoint_str = f"{feature_type}_cnn_model_epoch_{epoch}.pth"
    checkpoint = torch.load(checkpoint_dir / checkpoint_str)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()

    criterion = nn.BCEWithLogitsLoss()
    running_loss = 0.0
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            logits = model(inputs)

            loss = criterion(logits, targets)
            running_loss += loss.item() * inputs.size(0)

            probs = torch.sigmoid(logits)
            all_outputs.append(probs.cpu())
            all_targets.append(targets.cpu())

    all_outputs = torch.cat(all_outputs).numpy()
    all_targets = torch.cat(all_targets).numpy()

    epoch_loss = running_loss / len(dataset)
    print(f"\nValidation BCE Loss: {epoch_loss:.4f}")

    # Threshold outputs to get binary predictions
    threshold = 0.5
    predicts = (all_outputs >= threshold).astype(int)

    save_report_and_metrics(all_targets, all_outputs, predicts, epoch_loss, feature_type, epoch, "mono")


if __name__ == "__main__":
    evaluation(feature_type="melspectrogram", epoch=89)