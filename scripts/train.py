import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import torchaudio.transforms as T
from torchvision.transforms import v2

from data.dataset import AudioDataset
from models.cnn import AudioCNN

from config import SAMPLE_RATE, DURATION, DATA_ROOT, LABEL_LIST
from scripts.eval import evaluation
from util import generate_test_path_and_labels, dropblock_scheduler, set_drop_prob, ShiftSpectrogramX, summix_batch, \
    show_spec

from pathlib import Path

import os

# === CONFIG ===
NUM_CLASSES = len(LABEL_LIST)
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(feature_type="melspectrogram", start_epoch=0):
    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent
    checkpoint_dir = root_dir / "outputs/checkpoints"

    print("Using device:", DEVICE)

    train_root = DATA_ROOT + "/Train_highpass"
    file_paths, labels = generate_test_path_and_labels(train_root)

    # === Dataset & Dataloader ===
    dataset = AudioDataset(
        file_paths=file_paths,
        labels=labels,
        sample_rate=SAMPLE_RATE,
        duration=DURATION,
        feature_type=feature_type
    )

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

    # === Initialize data augmentation ===
    time_shift = ShiftSpectrogramX(shift=(-258, 258))

    time_mask = T.TimeMasking(time_mask_param=70).to(DEVICE)
    freq_mask = T.FrequencyMasking(freq_mask_param=70).to(DEVICE)

    #cutmix = v2.CutMix(num_classes=NUM_CLASSES)
    #mixup = v2.MixUp(num_classes=NUM_CLASSES)

    # === Model, Loss, Optimizer ===
    model = AudioCNN(num_classes=NUM_CLASSES).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    # === Load checkpoint ===
    if start_epoch > 0:
        checkpoint_str = f"{feature_type}_cnn_model_epoch_{start_epoch}.pth"
        checkpoint = torch.load(checkpoint_dir / checkpoint_str)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        print(f"Resuming training from epoch {start_epoch}")
    # === Training loop ===
    for epoch in range(start_epoch, EPOCHS):
        current_prob = dropblock_scheduler(epoch, EPOCHS, start_prob=0.3, end_prob=0.3)
        set_drop_prob(model, current_prob)

        model.train()
        running_loss = 0.0

        for batch in dataloader:
            inputs, targets = batch
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            inputs = torch.stack([time_shift(spec) for spec in inputs])

            inputs, targets = summix_batch(inputs, targets)
            #inputs, targets = cutmix(inputs, targets)
            #inputs, targets = mixup(inputs, targets)
            inputs = torch.stack([freq_mask(time_mask(spec)) for spec in inputs])

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        scheduler.step()

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch [{epoch + 1}/{EPOCHS}] - Loss: {epoch_loss:.4f}")

        # === Save checkpoint in outputs/checkpoints/ ===
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_path = os.path.join(checkpoint_dir, f"{feature_type}_cnn_model_epoch_{epoch + 1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': epoch_loss,
        }, checkpoint_path)

        evaluation(feature_type, (epoch + 1))
    print("Training done!")


if __name__ == "__main__":
    train(feature_type="melspectrogram", start_epoch=0)