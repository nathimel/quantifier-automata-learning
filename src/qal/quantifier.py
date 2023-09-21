"""Module defining various quantifiers as (reference) semantic automata, for data-generation and comparison."""

import torch
from qal.pfa import PFA

BINARY_ALPHABET = [0,1]

class QPFA(PFA):

    """A QPFA (Quantifier PFA) is just a PFA with some default parameters, and models a semantic automaton.
    """

    def __init__(self, init: torch.Tensor, T: torch.Tensor, final: torch.Tensor) -> None:
        num_states, num_symbols, _ = T.size()
        alphabet = range(num_symbols)
        super().__init__(num_states, alphabet, init, T, final)


class DefaultQPFA(QPFA):
        
    def __init__(self, num_states: int) -> None:

        # Hard-code initial state as first state using one-hot
        init = torch.zeros(num_states)
        init[0] = 1.0

        # final state distribution is uniform
        final = torch.ones(num_states) / num_states 
        final = torch.ones(num_states)

        # state-transition distribution is uniform
        # Assume binary alphabet
        T = torch.ones([num_states, len(BINARY_ALPHABET), num_states])

        super().__init__(init, T, final)



every = PFA(
    num_states = 2,
    alphabet=[0,1],
    init = torch.tensor([1., 0.,]), # one-hot on first state
    T = torch.tensor([
        # q0
        [[0., 1.,], 
         [1., 0.,]],
        # q1
        [[1., 0.,],
         [0., 1.,]]
    ]),
    final = torch.tensor([1., 0.,]) # also one-hot on first state
)

some = PFA(
    num_states = 2,
    alphabet=[0,1],
    init = torch.tensor([1., 0.,]),
    T = torch.tensor([
        [[1., 0.,], 
         [0., 1.,]],
        [[0., 1.,],
         [0., 1.,]]
    ]),
    final = torch.tensor([0., 1.,])
)

quantifier_map = {
    "every": every,
    "some": some,
}