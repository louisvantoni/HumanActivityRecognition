from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchaudio.transforms as T

from config import LABEL_LIST, DATA_ROOT, SAMPLE_RATE, DURATION
from data.dataset import AudioDataset
from models.binary_cnn import AudioBinaryCNN
from scripts.binary_eval import binary_eval
from util import generate_test_path_and_labels, dropblock_scheduler, set_drop_prob, ShiftSpectrogramX

NUM_CLASSES = len(LABEL_LIST)
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def binary_train(feature_type="melspectrogram", start_epoch=0):
    print("Using device:", DEVICE)

    time_shift = ShiftSpectrogramX(shift=(-258, 258))

    time_mask = T.TimeMasking(time_mask_param=70).to(DEVICE)
    freq_mask = T.FrequencyMasking(freq_mask_param=70).to(DEVICE)

    models = [AudioBinaryCNN().to(DEVICE) for _ in range(NUM_CLASSES)]
    optimizers = [optim.Adam(model.parameters(), lr=LEARNING_RATE) for model in models]
    schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5) for optimizer in optimizers]
    criterion = nn.BCEWithLogitsLoss()

    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent
    checkpoint_dir = root_dir / "outputs/checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for class_idx in range(NUM_CLASSES):
        class_name = LABEL_LIST[class_idx]
        class_dir = checkpoint_dir / class_name

        if start_epoch > 0:
            checkpoint_path = class_dir / f"{feature_type}_binary_cnn_model_epoch_{start_epoch}.pth"
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
                models[class_idx].load_state_dict(checkpoint["model_state_dict"])
                optimizers[class_idx].load_state_dict(checkpoint["optimizer_state_dict"])
                print(f"Loaded checkpoint for class '{class_name}' from epoch {checkpoint['epoch']}")
            else:
                print(f"Warning: No checkpoint found for class '{class_name}' at epoch {start_epoch}. Starting fresh.")

    train_root = DATA_ROOT + "/Train_highpass"
    file_paths, labels = generate_test_path_and_labels(train_root)

    dataset = AudioDataset(
        file_paths=file_paths,
        labels=labels,
        sample_rate=SAMPLE_RATE,
        duration=DURATION,
        feature_type=feature_type
    )

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

    for epoch in range(start_epoch, EPOCHS):
        class_loss_sums = [0.0 for _ in range(NUM_CLASSES)]
        class_counts = [0 for _ in range(NUM_CLASSES)]

        for specs, labels in dataloader:
            labels = labels.to(DEVICE)

            specs = torch.stack([time_shift(spec) for spec in specs])

            specs = time_mask(specs)
            specs = freq_mask(specs)
            specs = specs.to(DEVICE)

            for class_idx in range(NUM_CLASSES):
                model = models[class_idx]
                current_prob = dropblock_scheduler(epoch, EPOCHS, start_prob=0.3, end_prob=0.3)
                set_drop_prob(model, current_prob)
                optimizer = optimizers[class_idx]

                logits = model(specs).squeeze()
                targets = labels[:, class_idx]

                loss = criterion(logits, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                class_loss_sums[class_idx] += loss.item()
                class_counts[class_idx] += 1
        for scheduler in schedulers:
            scheduler.step()

        print(f"Epoch {epoch + 1}/{EPOCHS}")
        for class_idx in range(NUM_CLASSES):
            avg_loss = class_loss_sums[class_idx] / class_counts[class_idx]
            print(f"  Class {LABEL_LIST[class_idx]} | Avg Loss: {avg_loss:.4f}")

        for class_idx in range(NUM_CLASSES):
            class_name = LABEL_LIST[class_idx]
            class_dir = checkpoint_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            save_path = class_dir / f"{feature_type}_binary_cnn_model_epoch_{epoch+1}.pth"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": models[class_idx].state_dict(),
                "optimizer_state_dict": optimizers[class_idx].state_dict(),
            }, save_path)

            print(f"Saved model for class '{class_name}' at epoch {epoch+1} -> {save_path}")

        binary_eval(feature_type, (epoch + 1))
    print("Training done.")

if __name__ == "__main__":
    binary_train(feature_type="melspectrogram", start_epoch=0)