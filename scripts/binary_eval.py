import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path

from config import LABEL_LIST, DATA_ROOT, SAMPLE_RATE, DURATION
from data.dataset import AudioDataset
from models.binary_cnn import AudioBinaryCNN
from util import generate_test_path_and_labels, save_report_and_metrics

# === CONFIG ===
NUM_CLASSES = len(LABEL_LIST)
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def binary_eval(feature_type="melspectrogram", epoch=1):

    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent
    checkpoint_root = root_dir / "outputs/checkpoints"

    print(f"Using device: {DEVICE}")

    test_root = DATA_ROOT + "/Test_highpass"
    file_paths, labels = generate_test_path_and_labels(test_root)

    dataset = AudioDataset(
        file_paths=file_paths,
        labels=labels,
        sample_rate=SAMPLE_RATE,
        duration=DURATION,
        feature_type=feature_type,

    )

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    # === Load models ===
    models = []
    """for class_idx in range(NUM_CLASSES):
        model = AudioBinaryCNN().to(DEVICE)
        checkpoint_path = checkpoint_root / f"{LABEL_LIST[class_idx]}" / f"{feature_type}_binary_cnn_model_epoch_{epoch}.pth"
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        models.append(model)"""

    model= AudioBinaryCNN().to(DEVICE)
    checkpoint= torch.load(r"C:\Users\louis\PycharmProjects\HumanActivityClassification\outputs\checkpoints\talking\melspectrogram_binary_cnn_model_epoch_88.pth", map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    models.append(model)
    model= AudioBinaryCNN().to(DEVICE)
    checkpoint= torch.load(r"C:\Users\louis\PycharmProjects\HumanActivityClassification\outputs\checkpoints\laughing\melspectrogram_binary_cnn_model_epoch_71.pth", map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    models.append(model)
    model= AudioBinaryCNN().to(DEVICE)
    checkpoint= torch.load(r"C:\Users\louis\PycharmProjects\HumanActivityClassification\outputs\checkpoints\coughing\melspectrogram_binary_cnn_model_epoch_8.pth", map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    models.append(model)
    model= AudioBinaryCNN().to(DEVICE)
    checkpoint= torch.load(r"C:\Users\louis\PycharmProjects\HumanActivityClassification\outputs\checkpoints\children\melspectrogram_binary_cnn_model_epoch_59.pth", map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    models.append(model)
    model= AudioBinaryCNN().to(DEVICE)
    checkpoint= torch.load(r"C:\Users\louis\PycharmProjects\HumanActivityClassification\outputs\checkpoints\walking\melspectrogram_binary_cnn_model_epoch_53.pth", map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    models.append(model)
    model= AudioBinaryCNN().to(DEVICE)
    checkpoint= torch.load(r"C:\Users\louis\PycharmProjects\HumanActivityClassification\outputs\checkpoints\bicycle\melspectrogram_binary_cnn_model_epoch_37.pth", map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    models.append(model)
    model= AudioBinaryCNN().to(DEVICE)
    checkpoint= torch.load(r"C:\Users\louis\PycharmProjects\HumanActivityClassification\outputs\checkpoints\skateboard\melspectrogram_binary_cnn_model_epoch_1.pth", map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    models.append(model)
    model= AudioBinaryCNN().to(DEVICE)
    checkpoint= torch.load(r"C:\Users\louis\PycharmProjects\HumanActivityClassification\outputs\checkpoints\music\melspectrogram_binary_cnn_model_epoch_86.pth", map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    models.append(model)
    print("Loaded all models.")

    # Evaluation
    criterion = nn.BCEWithLogitsLoss()
    class_loss_sums = [0.0 for _ in range(NUM_CLASSES)]
    class_counts = [0 for _ in range(NUM_CLASSES)]
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for spec, labels in dataloader:
            spec = spec.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs_batch = []

            for class_idx in range(NUM_CLASSES):
                model = models[class_idx]
                logits = model(spec).squeeze()

                targets = labels[:, class_idx]

                loss = criterion(logits, targets)
                class_loss_sums[class_idx] += loss.item()
                class_counts[class_idx] += 1

                probs = torch.sigmoid(logits)
                outputs_batch.append(probs.cpu().numpy())

            outputs_batch = np.stack(outputs_batch, axis=1)
            all_outputs.append(outputs_batch)
            all_targets.append(labels.cpu().numpy())

    all_outputs = np.concatenate(all_outputs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    sum_avg_loss = 0.0
    for class_idx in range(NUM_CLASSES):
        avg_loss = class_loss_sums[class_idx] / class_counts[class_idx]
        sum_avg_loss += avg_loss
        print(f"  Class {LABEL_LIST[class_idx]} | Avg Loss: {avg_loss:.4f}")
    sum_avg_loss = sum_avg_loss/NUM_CLASSES

    # === Threshold ===
    threshold = 0.5
    predicts = (all_outputs >= threshold).astype(int)

    save_report_and_metrics(all_targets, all_outputs, predicts, sum_avg_loss, feature_type, epoch, "binary")

if __name__ == "__main__":
    binary_eval(feature_type="melspectrogram", epoch=101)