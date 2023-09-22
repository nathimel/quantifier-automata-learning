"""Various training-specific helpers, wrappers"""

import torch
from torch.utils.data import DataLoader

def get_optimizer(name: str) -> torch.optim.Optimizer:
    return {
        "Adam": torch.optim.Adam,
        "SGD": torch.optim.SGD,
    }[name]


class Trainer:

    def __init__(self, model, config) -> None:
        self.model = model
        # Loss function 
        # (more generally, we can try only using positive examples, and minimizing the NLL)
        self.criterion = torch.nn.BCEWithLogitsLoss()

        # Optimizer
        self.optimizer = get_optimizer(config.learning.optimizer)(params=model.parameters(), lr=config.learning.learning_rate)

    def train(self, dataloader: DataLoader) -> tuple[torch.Tensor]:
        train_loss = 0
        self.model.train()
        for _, batch in enumerate(dataloader):
            # Unpack batch (padded_seqs, seq_lengths, targets)
            seqs, seq_lengths, y = batch
            # Get PFA prediction error
            preds = self.model(seqs, seq_lengths)
            loss = self.criterion(preds, y)
            # Record loss
            train_loss += loss.item()
            # Update params
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        train_accuracy = 0. # TODO: implement

        return train_loss, train_accuracy
    
    def test(self, dataloader: DataLoader) -> tuple[torch.Tensor]:
        test_loss = 0
        with torch.no_grad():
            for _, batch in enumerate(dataloader):
                seqs, seq_lengths, y = batch
                preds = self.model(seqs, seq_lengths)
                loss = self.criterion(preds, y)
                test_loss += loss.item()

        test_accuracy = 0.
        
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