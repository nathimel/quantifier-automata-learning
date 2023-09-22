import torch
import pandas as pd

from torch.utils.data import DataLoader
from qal.quantifier import quantifier_map
from qal.data_gen import QuantifierStringDataset, custom_collate_fn
from qal.training import Trainer

from tests.test_quantifier_automata import zero, one

class TestTraining:

    def test_accuracy(self):

        # Define quantifier and dataset
        model = quantifier_map["every"]

        # Assumes exists, otherwise you need to generate first
        quantifier_data_fn = "data/quantifier_data/10.csv"
        quantifier_data = pd.read_csv(quantifier_data_fn)

        dataset = QuantifierStringDataset("every", quantifier_data, balanced=True)
        dataloader = DataLoader(dataset, collate_fn=custom_collate_fn)

        trainer = Trainer(model)
        trainer.threshold = 1.0 # since we have reference automaton

        test_loss, test_accuracy = trainer.test(dataloader)

        assert torch.isclose(torch.Tensor([test_loss]), zero)
        assert test_accuracy == 1.0
