"""Module defining various quantifiers as (reference) semantic automata, for data-generation and comparison."""

import torch
from qal.pfa import PFAModel

BINARY_ALPHABET = [0,1]

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



quantifier_map = {
    "every": every,
    "some": some,
    "at_least_three": at_least_three,
}