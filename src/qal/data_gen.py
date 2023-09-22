"""Generating labeled data for training and evaluating PFAs."""

import itertools
import torch
import pandas as pd

from qal.quantifier import quantifier_map
from typing import Generator

from torch.utils.data import Dataset

def string_to_int_tuple(string: str) -> tuple[int]:
    return tuple([int(x) for x in string])

def get_quantifier_labeled_data(data: pd.DataFrame, quant_name: str) -> pd.DataFrame:
    """Add a column for a quantifier of True/False labels, which labels strings True if they are accepted by the quantifier's automaton, and False otherwise."""

    # load pfa
    qpfa = quantifier_map[quant_name]
    data[quant_name] = None

    # ensure data is in string format
    data["string"] = data["string"].astype(str)

    labels = [
        True 
        if qpfa.forward_algorithm(string_to_int_tuple(row["string"])) == 0. else False 
        for _, row in data.iterrows()
    ]

    data[quant_name] = labels
    return data

def generate_up_to_length(max_length: int):
    """Generate binary strings up to a length."""
    return sorted(list(set(itertools.chain(*[list(generate_binary_strings(n)) for n in range(1, max_length)]))))

def generate_binary_strings(length: int) -> Generator:
    """Generate binary strings of a length."""
    for i in itertools.product("01", repeat=length):
        yield "".join(i)

class QuantifierStringDataset(Dataset):

    def __init__(self, quantifier_name: str, string_df: pd.DataFrame, balanced = True) -> None:
        """Construct a pytorch Dataset from a dataframe and column name.

        Args:
            quantifier_name: the column name of the dataframe corresponding to labels for that quantifier

            string_df: a pd.Dataframe containing binary strings and quantifier labels.

            balanced: whether to balance the dataset by downsampling instances of the majority label.
        """
        if balanced:
            positive_examples = string_df[string_df[quantifier_name] == True]
            negative_examples = string_df[string_df[quantifier_name] == False]
            min_class_size = min(len(positive_examples), len(negative_examples))

            # Undersample the majority class to match size of majority class
            balanced_dataset = pd.concat([
                positive_examples.sample(min_class_size, replace=False), negative_examples.sample(min_class_size, replace=False)
                ])
            string_df = balanced_dataset

        # List of tuples of ints
        self.strings = [string_to_int_tuple(x) for x in string_df["string"].astype(str).tolist()]

        # Note these are probabilities, not log probabilities
        self.labels = [torch.Tensor([x]) for x in string_df[quantifier_name].tolist()]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index) -> tuple[tuple[int], torch.Tensor]:
        return self.strings[index], self.labels[index]


def custom_collate_fn(samples: list[tuple[int]], pad_value=2):
    """
    Collate function to pad sequences in a sample to the same length.

    Args:
        samples (list): A list of binary strings
        pad_value (int, optional): The value to use for padding. Defaults to 2.

    Returns:
        padded_batch (tuple): A tuple containing two Tensors:
            - A Tensor of shape (batch_size, max_sequence_length) containing the padded sequences.
            - A Tensor of shape (batch_size, sequence_lengths) containing the actual sequence lengths.
    """
    # Find the maximum sequence length in the batch
    max_sequence_length = max(len(seq) for (seq, _) in samples)

    padded_seqs = []
    seq_lengths = []
    targets = []
    # Pad each sequence to the maximum length and record their original lengths
    for (seq, target) in samples:
        seq_length = len(seq)
        pad_length = max_sequence_length - seq_length
        padded_seq = seq + tuple([pad_value]) * pad_length
        padded_seqs.append(padded_seq)
        seq_lengths.append(seq_length)
        targets.append(target)

    # Convert the lists of padded sequences and lengths to PyTorch Tensors
    padded_seqs = torch.tensor(padded_seqs)
    seq_lengths = torch.tensor(seq_lengths)
    targets = torch.tensor(targets)

    return padded_seqs, seq_lengths, targets
