from pathlib import Path

import librosa
import numpy
import torch

from models.binary_cnn import AudioBinaryCNN
from config import LABEL_LIST, NOISE_FILE
from util import wiener_filter_denoise

NUM_CLASSES = len(LABEL_LIST)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def binary_predict(audio_path, epoch, threshold=0.5, feature_type="melspectrogram"):
    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent
    checkpoint_root = root_dir / "outputs/checkpoints"

    # Load models
    models = []
    for class_idx in range(NUM_CLASSES):
        model = AudioBinaryCNN().to(DEVICE)
        checkpoint_path = checkpoint_root / f"{LABEL_LIST[class_idx]}" / f"{feature_type}_binary_cnn_model_epoch_{epoch}.pth"
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        models.append(model)

    print("Loaded all models.")

    audio, _ = librosa.load(audio_path)
    feature = wiener_filter_denoise(audio, NOISE_FILE)
    feature = (feature - feature.min()) / (feature.max() - feature.min())
    feature = torch.tensor(feature).unsqueeze(0).unsqueeze(0).float()
    feature = feature.to(DEVICE)
    probability_vec = numpy.zeros(8)

    with torch.no_grad():
        for class_idx in range(NUM_CLASSES):
            model = models[class_idx]
            logit = model(feature).squeeze()
            prob = torch.sigmoid(logit)
            probability_vec[class_idx] = prob

    # Thresholding
    predicts = (probability_vec >= threshold).astype(int)

    return predicts

if __name__ == "__main__":
    file_path = r"C:\Users\louis\PycharmProjects\HumanActivityClassification\data\processed\HumanActivityAudiodata_normalized\Test\21052015_ZOOM0008_Tr34-58-1238.wav"
    output = binary_predict(file_path, 64)
    print(output)