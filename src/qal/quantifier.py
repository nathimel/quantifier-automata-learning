"""Module defining various quantifiers as (reference) semantic automata, for data-generation and comparison."""

import torch
from qal.pfa import PFAModel

BINARY_ALPHABET = [0,1]

# class QPFA(PFA):

#     """A QPFA (Quantifier PFA) is just a PFA with some default parameters, and models a semantic automaton.
#     """

#     def __init__(self, T: torch.Tensor, final: torch.Tensor) -> None:
#         num_states, num_symbols, _ = T.size()

#         # assume first state is always initial
#         init = torch.zeros(num_states)
#         init[0] = 1.

#         alphabet = range(num_symbols) # 2

#         super().__init__(num_states, alphabet, init, T, final)


# Dummy initialize
every = PFAModel(
    num_states = 2,
    alphabet = BINARY_ALPHABET,
)
# Hard-code parameters
T = torch.tensor([
    # q0
    [[0., 1.,], 
        [1., 0.,]],
    # q1
    [[0., 1.,],
    [0., 1.,]]
])
f = torch.tensor([1., 0.,]) # one-hot on last state
every.transition.T_logits = torch.nn.Parameter(torch.log(T))
every.final.f_logits = torch.nn.Parameter(torch.log(f))

# every = PFA(
#     num_states = 2,
#     alphabet=[0,1],
#     init = torch.tensor([1., 0.,]), # one-hot on first state
# ,
#     final = torch.tensor([1., 0.,]) # also one-hot on first state
# )

some = PFAModel(
    num_states = 2,
    alphabet = BINARY_ALPHABET,
)
T = torch.tensor([
        [[1., 0.,], 
         [0., 1.,]],
        [[0., 1.,],
         [0., 1.,]]
])
f = torch.tensor([0., 1.,])
some.transition.T_logits = torch.nn.Parameter(torch.log(T))
some.final.f_logits = torch.nn.Parameter(torch.log(f))

# some = PFA(
#     num_states = 2,
#     alphabet=[0,1],
#     init = torch.tensor([1., 0.,]),
#     T = torch.tensor([
#         [[1., 0.,], 
#          [0., 1.,]],
#         [[0., 1.,],
#          [0., 1.,]]
#     ]),
#     final = torch.tensor([0., 1.,])
# )

at_least_three = PFAModel(
    num_states = 4,
    alphabet = BINARY_ALPHABET,
)
T = torch.tensor([
        [[1., 0., 0., 0.], 
         [0., 1., 0., 0.]],
        [[0., 1., 0., 0.], 
         [0., 0., 1., 0.]],
        [[0., 0., 1., 0.], 
         [0., 0., 0., 1.]],
        [[0., 0., 0., 1.], 
         [0., 0., 0., 1.,]],
])
f = torch.tensor([0., 0., 0., 1.,])
at_least_three.transition.T_logits = torch.nn.Parameter(torch.log(T))
at_least_three.final.f_logits = torch.nn.Parameter(torch.log(f))

# at_least_three = PFA(
#     num_states = 4,
#     alphabet=[0,1],
#     init = torch.tensor([1., 0., 0., 0.,]),
#     T = torch.tensor([
#         [[1., 0., 0., 0.], 
#          [0., 1., 0., 0.]],
#         [[0., 1., 0., 0.], 
#          [0., 0., 1., 0.]],
#         [[0., 0., 1., 0.], 
#          [0., 0., 0., 1.]],
#         [[0., 0., 0., 1.], 
#          [0., 0., 0., 1.,]],
#     ]),
#     final = torch.tensor([0., 0., 0., 1.,])
# )

quantifier_map = {
    "every": every,
    "some": some,
    "at_least_three": at_least_three,
}