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
every.T_logits = torch.nn.Parameter(torch.log(T))
every.f_logits = torch.nn.Parameter(torch.log(f))

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
some.T_logits = torch.nn.Parameter(torch.log(T))
some.f_logits = torch.nn.Parameter(torch.log(f))

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
         [0., 0., 0., 1.]],
])
f = torch.tensor([0., 0., 0., 1.])
at_least_three.T_logits = torch.nn.Parameter(torch.log(T))
at_least_three.f_logits = torch.nn.Parameter(torch.log(f))


at_least_four = PFAModel(
    num_states = 5,
    alphabet = BINARY_ALPHABET,
)
T = torch.tensor([
        [[1., 0., 0., 0., 0.], 
         [0., 1., 0., 0., 0.]],
        [[0., 1., 0., 0., 0.], 
         [0., 0., 1., 0., 0.]],
        [[0., 0., 1., 0., 0.], 
         [0., 0., 0., 1., 0.]],
        [[0., 0., 0., 1., 0.], 
         [0., 0., 0., 0., 1.]],
        [[0., 0., 0., 0., 1.], 
         [0., 0., 0., 0., 1.]]
])
f = torch.tensor([0., 0., 0., 0., 1.,])
at_least_four.T_logits = torch.nn.Parameter(torch.log(T))
at_least_four.f_logits = torch.nn.Parameter(torch.log(f))

at_least_six_or_at_most_two = PFAModel(7, BINARY_ALPHABET)
T = torch.tensor([
        [[1., 0., 0., 0., 0., 0., 0.], 
         [0., 1., 0., 0., 0., 0., 0.]],
        [[0., 1., 0., 0., 0., 0., 0.], 
         [0., 0., 1., 0., 0., 0., 0.]],
        [[0., 0., 1., 0., 0., 0., 0.], 
         [0., 0., 0., 1., 0., 0., 0.]],
        [[0., 0., 0., 1., 0., 0., 0.], 
         [0., 0., 0., 0., 1., 0., 0.]],
        [[0., 0., 0., 0., 1., 0., 0.], 
         [0., 0., 0., 0., 0., 1., 0.]],
        [[0., 0., 0., 0., 0., 1., 0.], 
         [0., 0., 0., 0., 0., 0., 1.]],
        [[0., 0., 0., 0., 0., 0., 1.], 
         [0., 0., 0., 0., 0., 0., 1.]]        
])
f = torch.tensor([1., 1., 1., 0., 0., 0., 1.])
at_least_six_or_at_most_two.T_logits = torch.nn.Parameter(torch.log(T))
at_least_six_or_at_most_two.f_logits = torch.nn.Parameter(torch.log(f))

quantifier_map = {
    "every": every,
    "some": some,
    "at_least_three": at_least_three,
    "at_least_four": at_least_four,
    "at_least_six_or_at_most_two": at_least_six_or_at_most_two,
}