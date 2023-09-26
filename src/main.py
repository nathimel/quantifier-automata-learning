"""Main experiment script that trains and evaluates pfa learning."""

import hydra
import os
import torch

import pandas as pd

from qal import util
from qal.training import Trainer, EarlyStopper
from qal.quantifier import BINARY_ALPHABET
from qal.pfa import PFAModel
from qal.data_gen import QuantifierStringDataset, custom_collate_fn

from torch.utils.data import DataLoader, random_split

from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    util.set_seed(config.seed)

    quantifier_data_fn = util.get_quantifier_data_fn(config)
    quantifier_name = config.quantifier.name
    curves_fn = config.filepaths.curves_fn

    epochs = int(config.learning.epochs)
    init_temperature = config.learning.init_temperature
    batch_size = config.learning.batch_size
    split = config.learning.data.train_test_split
    balanced = config.learning.data.balanced
    early_stopping = config.learning.early_stopping

    verbose = config.verbose
    print_freq = config.learning.print_frequency

    # Dataset
    dataset = QuantifierStringDataset(
        quantifier_name, 
        pd.read_csv(quantifier_data_fn),
        balanced=balanced,
        max_size=config.learning.data.max_size,
        )
    train_size = int(split * len(dataset))
    test_size = len(dataset) - train_size
    training_data, test_data = random_split(dataset, [train_size, test_size])

    dataloader_kwargs = {
        "batch_size": batch_size, 
        "collate_fn": custom_collate_fn,
        "shuffle": True,
    }

    train_dataloader = DataLoader(
        training_data, 
        **dataloader_kwargs,
        )
    test_dataloader = DataLoader(
        test_data,
        **dataloader_kwargs,
    )
    if verbose:
        print(f"Training data size = {len(training_data)}")
        print(f"Test data size = {len(test_data)}")

    # Set device
    # device = torch.device("mps") # for some reason slower
    device = torch.device("cpu")

    # Quantifier PFA learning model
    model = PFAModel(
        num_states=config.learning.num_states,
        alphabet=BINARY_ALPHABET,
        init_temperature=init_temperature,
    )
    model.to(device)
    
    trainer = Trainer(model, config, device)
    if early_stopping:
        early_stopper = EarlyStopper(
            patience=config.learning.patience,
            accuracy_threshold=config.learning.accuracy_threshold,
            loss_threshold=config.learning.loss_threshold,
        )

    # Main training loop
    curves = {
        "train_losses": [],
        "train_accuracies": [],
        "test_losses": [],
        "test_accuracies": [],
    }

    for epoch in range(epochs):
        if verbose and epoch % print_freq == 0:
            print(f"Epoch {epoch}/{epochs}\n-------------------------------")

        # Train
        train_loss, train_accuracy = trainer.train(train_dataloader)

        # Track training progress
        avg_train_loss = train_loss / len(train_dataloader)
        if verbose and (epoch) % print_freq == 0:
            print(f"avg train loss: {avg_train_loss:>7f}")
            print(f"avg train accuracy: {train_accuracy:>7f}")
        curves["train_losses"].append(avg_train_loss)
        curves["train_accuracies"].append(train_accuracy)

        # Test
        test_loss, test_accuracy = trainer.test(test_dataloader)

        # Track test progress
        avg_test_loss = test_loss / len(test_dataloader)
        if verbose and (epoch) % print_freq == 0:
            print(f"avg test loss: {avg_test_loss:>7f}")
            print(f"avg test accuracy: {test_accuracy:>7f}")
        curves["test_losses"].append(avg_test_loss)
        curves["test_accuracies"].append(test_accuracy)

        # Model checkpoint
        if (epoch + 1) % config.learning.checkpoint_freq == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'avg_loss': avg_test_loss,
            }
            print("Saving model checkpoint.")
            torch.save(checkpoint, config.filepaths.checkpoint_fn)
            print(f"Wrote checkpoint to {os.path.join(os.getcwd(), config.filepaths.checkpoint_fn)}")            
            util.save_curves(curves, curves_fn)

        # Check if early stopping criteria met
        if early_stopping and early_stopper.should_stop(avg_test_loss, test_accuracy):
            print(f"Early stopping after {epoch+1} epochs.")
            break

        if verbose and (epoch) % print_freq == 0:
            print("-------------------------------")

    # Save curves
    util.save_curves(curves, curves_fn)

    # Save final model
    torch.save(model.state_dict(), config.filepaths.model_fn)
    print(f"Wrote model to {os.path.join(os.getcwd(), config.filepaths.model_fn)}")


if __name__ == "__main__":
    main()