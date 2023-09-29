"""Module defining various quantifiers as (reference) semantic automata, for data-generation and comparison."""

import torch
from qal.pfa import PFAModel

BINARY_ALPHABET = [0,1]

##############################################################################
# Test quantifiers
##############################################################################

# Every
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


# Some
T = torch.tensor([
        [[1., 0.,], 
         [0., 1.,]],
        [[0., 1.,],
         [0., 1.,]]
])
f = torch.tensor([0., 1.,])
some = PFAModel.from_probs(T, f)


# At least three
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

# At least four
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

# At least six (re-use T above)
f = torch.tensor([0., 0., 0., 0., 0., 0., 1.])
at_least_six = PFAModel.from_probs(T, f)

# At most three
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

# First three
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

# Even
T = torch.tensor([
    [[1., 0.,],
    [0., 1.,]],    
    [[0., 1.,], 
     [1., 0.,]]
])
f = torch.tensor([1., 0.])
even = PFAModel.from_probs(T, f)

# Exactly two (differs from at least 3 only by final state)
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
f = torch.tensor([0., 0., 1., 0.])
exactly_two = PFAModel.from_probs(T, f)

##############################################################################
# End quantifiers
##############################################################################

quantifier_map = {
    "every": every,
    "some": some,
    "at_least_three": at_least_three,
    "at_least_four": at_least_four,
    "at_least_six": at_least_six,
    "at_least_six_or_at_most_two": at_least_six_or_at_most_two,
    "at_most_three": at_most_three,
    "first_three": first_three,
    "even": even,
    "exactly_two": exactly_two,
}