"""Module defining various quantifiers as (reference) semantic automata, for data-generation and comparison."""

import torch
from qal.pfa import PFAModel

BINARY_ALPHABET = [0,1]

##############################################################################
# Test quantifiers
##############################################################################

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
every = PFAModel.from_probs(T, f)


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
some = PFAModel.from_probs(T, f)


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
at_least_three = PFAModel.from_probs(T, f)


##############################################################################
# Experiment quantifiers
##############################################################################

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
at_least_four = PFAModel.from_probs(T, f)

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
at_least_six_or_at_most_two = PFAModel.from_probs(T, f)


at_most_three = PFAModel(5, BINARY_ALPHABET)
# same as at_least_four
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
f = torch.tensor([1., 1., 1., 1., 0.])
at_most_three = PFAModel.from_probs(T, f)


first_three = PFAModel(5, BINARY_ALPHABET)
T = torch.tensor([
        [[0., 0., 0., 0., 1.], 
         [0., 1., 0., 0., 0.]],
        [[0., 0., 0., 0., 1.], 
         [0., 0., 1., 0., 0.]],
        [[0., 0., 0., 0., 1.], 
         [0., 0., 0., 1., 0.]],
        [[0., 0., 0., 1., 0.], 
         [0., 0., 0., 1., 0.]],
        [[0., 0., 0., 0., 1.], 
         [0., 0., 0., 0., 1.]],
])
f = torch.tensor([0., 0., 0., 1., 0.])
first_three = PFAModel.from_probs(T, f)


even = PFAModel(2, BINARY_ALPHABET)
T = torch.tensor([
    [[1., 0.,],
    [0., 1.,]],    
    [[0., 1.,], 
     [1., 0.,]]
])
f = torch.tensor([1., 0.])
even = PFAModel.from_probs(T, f)

##############################################################################
# End quantifiers
##############################################################################

quantifier_map = {
    "every": every,
    "some": some,
    "at_least_three": at_least_three,
    "at_least_four": at_least_four,
    "at_least_six_or_at_most_two": at_least_six_or_at_most_two,
    "at_most_three": at_most_three,
    "first_three": first_three,
    "even": even,
}