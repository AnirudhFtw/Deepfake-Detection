import os
import random
import numpy as np
import torch


def seed_everything(seed=42):
    """
    Set random seeds for reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """
    Keeps track of average values like loss or accuracy.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, value, n=1):
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_accuracy(outputs, labels):
    """
    Computes batch accuracy.
    """

    _, preds = torch.max(outputs, dim=1)

    correct = (preds == labels).sum().item()

    accuracy = correct / labels.size(0)

    return accuracy


def save_checkpoint(model, optimizer, epoch, best_acc, path):
    """
    Save model checkpoint.
    """

    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_acc": best_acc
    }

    torch.save(checkpoint, path)


def load_checkpoint(model, optimizer, path, device):
    """
    Load saved checkpoint.
    """

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint["epoch"]

    best_acc = checkpoint["best_acc"]

    return model, optimizer, epoch, best_acc


class EarlyStopping:
    """
    Stops training if validation accuracy
    doesn't improve for 'patience' epochs.
    """

    def __init__(self, patience=7):

        self.patience = patience

        self.best_score = None

        self.counter = 0

        self.stop = False

    def __call__(self, score):

        if self.best_score is None:

            self.best_score = score

            return

        if score > self.best_score:

            self.best_score = score

            self.counter = 0

        else:

            self.counter += 1

            if self.counter >= self.patience:

                self.stop = True