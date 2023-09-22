"""Various training-specific helpers, wrappers"""

import torch
from torch.utils.data import DataLoader
from qal.pfa import PFAModel

def get_optimizer(name: str) -> torch.optim.Optimizer:
    return {
        "Adam": torch.optim.Adam,
        "SGD": torch.optim.SGD,
    }[name]


class Trainer:

    def __init__(self, model: PFAModel, config = None, device = None) -> None:
        """Initialize a Trainer to encapsulate training, testing of the model."""
        # Critical stuff

        self.model = model
        # loss function (more generally, we can try only using positive examples, and minimizing the NLL)
        self.criterion = torch.nn.BCELoss()

        # Extra stuff

        self.device = torch.device("cpu")
        if device is not None:
            self.device = device

        # threshold for predicting True/False in our binary classification
        self.threshold = None
        self.optimizer = None

        if config is not None:
            self.optimizer = get_optimizer(config.learning.optimizer)(params=model.parameters(), lr=config.learning.learning_rate)

            self.threshold = config.learning.threshold

    def train(self, dataloader: DataLoader) -> tuple[torch.Tensor]:
        train_loss = 0
        total_correct = 0
        total_samples = 0
        self.model.train()
        for _, batch in enumerate(dataloader):
            # Unpack batch padded to max_length
            seqs, seq_lengths, targets = batch
            seqs = seqs.to(self.device)
            seq_lengths = seq_lengths.to(self.device)
            targets = targets.to(self.device)
            # Get PFA prediction error
            outputs = self.model(seqs, seq_lengths)
            loss = self.criterion(outputs, targets)
            # try:
            #     loss = self.criterion(outputs, targets)
            # except RuntimeError:
            #     breakpoint()
            # Record loss
            train_loss += loss.item()
            # Update params
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Calculate training accuracy
            predicted_labels = (outputs >= self.threshold).float()
            correct = (predicted_labels == targets).sum().item()
            total_correct += correct
            total_samples += len(targets)
        train_accuracy = total_correct / total_samples

        return train_loss, train_accuracy
    
    def test(self, dataloader: DataLoader) -> tuple[torch.Tensor]:
        test_loss = 0
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for _, batch in enumerate(dataloader):
                seqs, seq_lengths, targets = batch
                seqs = seqs.to(self.device)
                seq_lengths = seq_lengths.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(seqs, seq_lengths)
                loss = self.criterion(outputs, targets)
                # try:
                #     loss = self.criterion(outputs, targets)
                # except RuntimeError:
                #     breakpoint()

                test_loss += loss.item()

                # Calculate training accuracy
                predicted_labels = (outputs >= self.threshold).float()
                correct = (predicted_labels == targets).sum().item()
                total_correct += correct
                total_samples += len(targets)
        test_accuracy = total_correct / total_samples
        
        return test_loss, test_accuracy


class EarlyStopper:
    def __init__(self, patience: int = 100):
        """Initialize an EarlyStopper.

        Args:
            patience: how many consecutive epochs with no improvement should trigger early stopping.
        """
        self.patience = patience
        self.counter = 0
        self.best_val_loss = torch.inf

    def should_stop(self, val_loss: torch.Tensor) -> bool:
        """Return True if `val_loss` has not improved after `patience` consecutive epochs.

        Args:
            val_loss: the average validation loss of the model over one epoch.
        """
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience