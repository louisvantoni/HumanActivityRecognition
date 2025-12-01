import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import v2
import torchaudio.transforms as T

from data.dataset import AudioDataset
from models.cnn import AudioCNN

from config import SAMPLE_RATE, DURATION, DATA_ROOT, LABEL_LIST
from scripts.finetune_eval import finetune_eval
from util import generate_test_path_and_labels, dropblock_scheduler, set_drop_prob

from pathlib import Path
import os

# === CONFIG ===
NUM_CLASSES = len(LABEL_LIST)
BATCH_SIZE = 32
FINETUNE_EPOCHS = 30
BASE_LR = 1e-5
HEAD_LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def finetune(feature_type="melspectrogram", checkpoint_epoch=None, unfreeze_interval=10):
    """
    Fine-tune a pretrained CNN with progressive unfreezing.
    Args:
        feature_type (str): Feature type, e.g. "melspectrogram"
        checkpoint_epoch (int): Which checkpoint epoch to load
        unfreeze_interval (int): How many epochs before unfreezing more layers
    """
    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent
    checkpoint_dir = root_dir / "outputs/checkpoints"

    print("Using device:", DEVICE)

    train_root = DATA_ROOT + "/Train"
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
    #time_mask = T.TimeMasking(time_mask_param=80).to(DEVICE)
    #freq_mask = T.FrequencyMasking(freq_mask_param=80).to(DEVICE)

    #cutmix = v2.CutMix(num_classes=NUM_CLASSES)
    #mixup = v2.MixUp(num_classes=NUM_CLASSES)
    #cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

    # === Model ===
    model = AudioCNN(num_classes=NUM_CLASSES).to(DEVICE)

    # === Load pretrained checkpoint ===
    assert checkpoint_epoch is not None, "Please specify checkpoint_epoch for fine-tuning."
    checkpoint_str = f"{feature_type}_cnn_model_epoch_{checkpoint_epoch}.pth"
    checkpoint = torch.load(checkpoint_dir / checkpoint_str, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    print(f"Loaded pretrained checkpoint from epoch {checkpoint_epoch}")

    # === Initially freeze all convolutional Layers ===
    model.features.requires_grad = False


    trainable_layers = ["fc"]

    criterion = nn.BCEWithLogitsLoss()

    # === Fine-tuning loop ===
    for epoch in range(FINETUNE_EPOCHS):
        # === Progressive unfreezing ===
        if (epoch + 1) % unfreeze_interval == 0:
            # Unfreeze one more layer (from the end of the model backwards)
            frozen_layers = [n for n, p in model.named_parameters() if not p.requires_grad]
            if frozen_layers:
                layer_to_unfreeze = frozen_layers[-1]
                for n, p in model.named_parameters():
                    if n == layer_to_unfreeze:
                        p.requires_grad = True
                        trainable_layers.append(layer_to_unfreeze)
                        print(f"Epoch {epoch+1}: unfroze layer {layer_to_unfreeze}")
                        break

        # Update optimizer with new set of trainable params
        optimizer = optim.Adam(
            [
                {"params": [p for n, p in model.named_parameters() if p.requires_grad and "features" in n], "lr": BASE_LR},
                {"params": [p for n, p in model.named_parameters() if p.requires_grad and "fc" in n], "lr": HEAD_LR},
            ],
            weight_decay=0.0001
        )

        current_prob = dropblock_scheduler(epoch, FINETUNE_EPOCHS, start_prob=0.0, end_prob=0.1)
        set_drop_prob(model, current_prob)

        # === Training ===
        model.train()
        running_loss = 0.0
        for batch in dataloader:
            inputs, targets = batch
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            #inputs, targets = cutmix_or_mixup(inputs, targets)
            #inputs = torch.stack([freq_mask(time_mask(spec)) for spec in inputs])

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(dataset)
        print(f"Fine-tune Epoch [{epoch + 1}/{FINETUNE_EPOCHS}] - Loss: {epoch_loss:.4f} | Trainable: {len(trainable_layers)} layers")

        # === Save checkpoint ===
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"{feature_type}_cnn_finetuned_epoch_{epoch + 1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
        }, checkpoint_path)

        finetune_eval(feature_type, (epoch + 1))

    print("Progressive fine-tuning complete!")


if __name__ == "__main__":
    finetune(feature_type="melspectrogram", checkpoint_epoch=95, unfreeze_interval=10)