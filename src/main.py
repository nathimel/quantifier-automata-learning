"""Main experiment script that trains and evaluates pfa learning."""

import hydra
import os
import torch

import pandas as pd

from qal import util
from qal import training
from qal.quantifier import BINARY_ALPHABET
from qal.pfa import PFAModel
from qal.data_gen import QuantifierStringDataset, custom_collate_fn

from torch.utils.data import DataLoader, random_split

from omegaconf import DictConfig

import sys


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    util.set_seed(config.seed)

    quantifier_data_fn = util.get_quantifier_data_fn(config)
    quantifier_name = config.quantifier.name

    epochs = int(config.learning.epochs)
    num_states = config.learning.num_states
    init_temperature = config.learning.init_temperature
    batch_size = config.learning.batch_size
    split = config.learning.data.train_test_split
    balanced = config.learning.data.balanced

    verbose = config.verbose

    # Dataset
    dataset = QuantifierStringDataset(
        quantifier_name, 
        pd.read_csv(quantifier_data_fn),
        balanced=balanced,
        )
    train_size = int(split * len(dataset))
    test_size = len(dataset) - train_size
    training_data, test_data = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(
        training_data, 
        batch_size=batch_size, 
        collate_fn=custom_collate_fn, 
        shuffle=True,
        )
    size = len(train_dataloader.dataset)

    # Quantifier PFA learning model
    model = PFAModel(
        num_states=num_states,
        alphabet=BINARY_ALPHABET,
        init_temperature=init_temperature,
    )

    # Loss function 
    # (more generally, we can try only using positive examples, and minimizing the NLL)
    criterion = torch.nn.BCELoss()

    # Optimizer
    optimizer = training.get_optimizer(config.learning.optimizer)
    opt = optimizer(params=model.parameters())

    # Main training loop
    running_loss = 0.
    for epoch in range(epochs):
        if verbose and epoch % 100 == 0:
            print(f"Epoch {epoch+1}\n-------------------------------")
        
        for batch_num, batch in enumerate(train_dataloader):
            
            # Unpack batch (padded_seqs, seq_lengths, targets)
            seqs, seq_lengths, y = batch

            # Get PFA prediction error
            preds = model(seqs, seq_lengths)
            try:
                loss = criterion(preds, y)
            except:
                breakpoint()

            # Record loss
            running_loss += loss.item() * seqs.size(0)

            # breakpoint()

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss = loss.item()
            current = batch_num * len(seqs)
            if verbose and epoch % 100 == 0:
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
        running_loss /= len(train_dataloader.sampler)

if __name__ == "__main__":
    main()