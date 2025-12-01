from scripts.eval import evaluation
from scripts.finetune import finetune

if __name__ == "__main__":
    for epoch in range(1,100):
        evaluation(feature_type="melspectrogram", epoch=epoch)
    finetune(feature_type="melspectrogram", checkpoint_epoch=94, unfreeze_interval=5)