"""Generating labeled data for training and evaluating PFAs."""

import itertools
import pandas as pd

from qal.quantifier import quantifier_map
from typing import Generator

def string_to_int_tuple(string: str) -> tuple[int]:
    return tuple([int(x) for x in string])

def get_quantifier_labeled_data(data: pd.DataFrame, quant_name: str) -> pd.DataFrame:
    """Add a column for a quantifier of True/False labels, which labels strings True if they are accepted by the quantifier's automaton, and False otherwise."""

    # load pfa
    qpfa = quantifier_map[quant_name]

    for _, row in data.iterrows():
        # breakpoint()
        x = string_to_int_tuple(row["string"])
        logp = qpfa.logp_string(x)
        label = True if logp == 0. else False # handles log0?
        print(x, logp, label)

def generate_up_to_length(max_length: int):
    """Generate binary strings up to a length."""
    return sorted(list(set(itertools.chain(*[list(generate_binary_strings(n)) for n in range(1, max_length)]))))

def generate_binary_strings(length: int) -> Generator:
    """Generate binary strings of a length."""
    for i in itertools.product("01", repeat=length):
        yield "".join(i)

