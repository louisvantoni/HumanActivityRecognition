from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import LABEL_LIST, DATA_ROOT, DURATION

from data.transfer_dataset import TransferAudioDataset
from external.audioset_tagging_cnn.pytorch.models import Cnn14
from scripts.transfer_eval import transfer_eval
from util import generate_test_path_and_labels

# === CONFIG ===
NUM_CLASSES = len(LABEL_LIST)
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PANN_SR = 32000
PANN_WS = 1024
PANN_HS = 320
PANN_MB = 64


def train_transfer(feature_type="melspectrogram", start_epoch=0):
    print(f"Using device: {DEVICE}")

    # === Model, Optimizer, Loss ===
    model = Cnn14(
        sample_rate=PANN_SR,
        window_size=1024,
        hop_size=320,
        mel_bins=64,
        fmin=50,
        fmax=14000,
        classes_num=527
    )

    pretrained_path = r"C:\Users\louis\PycharmProjects\HumanActivityClassification\external\Cnn14_mAP=0.431.pth"
    checkpoint = torch.load(pretrained_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model'], strict=False)

    in_features = model.fc_audioset.in_features
    model.fc_audioset = nn.Linear(in_features, NUM_CLASSES)

    model.to(DEVICE)

    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze classifier layers
    for param in model.fc1.parameters():
        param.requires_grad = True
    for param in model.fc_audioset.parameters():
        param.requires_grad = True

    # Only train fc1 and fc_audioset
    optimizer = optim.Adam(
        list(model.fc1.parameters()) + list(model.fc_audioset.parameters()),
        lr=LEARNING_RATE
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    criterion = nn.BCELoss()

    # === Checkpoint setup ===
    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent
    checkpoint_dir = root_dir / "outputs/checkpoints_transfer"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # === Dataset & DataLoader ===
    train_root = DATA_ROOT + "/Train_highpass"
    file_paths, labels = generate_test_path_and_labels(train_root)

    dataset = TransferAudioDataset(file_paths=file_paths, labels=labels, sample_rate=32000, duration=DURATION)

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )

    # === Training Loop ===
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        epoch_loss = 0.0

        for waveforms, labels_batch in dataloader:
            waveforms = waveforms.to(DEVICE)
            labels_batch = labels_batch.to(DEVICE)
            outputs = model(waveforms)
            probs = outputs["clipwise_output"]

            loss = criterion(probs, labels_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

        # Save checkpoint
        save_path = checkpoint_dir / f"{feature_type}_pann_transfer_epoch_{epoch+1}.pth"
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                'loss': epoch_loss,
            },
            save_path,
        )
        print(f"Saved checkpoint -> {save_path}")
        transfer_eval(feature_type=feature_type, epoch=(epoch + 1))

    print("Training finished.")


if __name__ == "__main__":
    train_transfer(feature_type="melspectrogram", start_epoch=0)