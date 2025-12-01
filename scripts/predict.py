from pathlib import Path

import librosa
import torch

from models.cnn import AudioCNN
from config import LABEL_LIST, DATA_ROOT, SAMPLE_RATE
from util import extract_melspectrogram, extract_mfcc

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict(audio, threshold=0.5, feature_type="melspectrogram"):
    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent
    checkpoint_dir = root_dir / "outputs/checkpoints"

    model_path = checkpoint_dir / f"{feature_type}_cnn_finetuned_epoch_49.pth"

    num_classes = len(LABEL_LIST)
    model = AudioCNN(num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    if feature_type == 'melspectrogram':
        feature = extract_melspectrogram(audio, sr)
        feature = (feature - feature.min()) / (feature.max() - feature.min())
    elif feature_type == 'mfcc':
        feature = extract_mfcc(audio, sr)
    else:
        raise ValueError(f"Invalid feature_type: {feature_type}")
    feature = torch.tensor(feature).unsqueeze(0).unsqueeze(0).float()
    feature = feature.to(DEVICE)

    with torch.no_grad():
        logits = model(feature)
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).int().cpu().numpy()[0]

    pred_labels = [LABEL_LIST[i] for i, val in enumerate(preds) if val == 1]

    return pred_labels, probs.cpu().numpy()[0]

if __name__ == "__main__":
    test_file = "/Test/21052015_ZOOM0008_Tr34-58-1238.wav"
    #test_file = "/Test/12052015ZOOM0007_Tr34-1-5.wav"
    test_file_path = DATA_ROOT + test_file

    test_audio, sr = librosa.load(test_file_path, sr=SAMPLE_RATE)

    pred_labels, probs = predict(test_audio, feature_type="melspectrogram")

    print(f"\nFile: {test_file}")
    print(f"Predicted labels: {pred_labels}")
    print(f"Raw probabilities: {probs}")