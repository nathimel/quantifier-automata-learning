"""Module defining various quantifiers as (reference) semantic automata, for data-generation and comparison."""

import torch
from qal.pfa import PFAModel

BINARY_ALPHABET = [0,1]

# TODO: make this a method of PFAModel
def set_pfa_params(pfa: PFAModel, T, f):
    """Takes stochastic matrix T and binary vector f and creates the appropriate corresponding logits."""
    # softmax is not invertible, but since we're creating delta functions, we can just choose very large (100) for 1, and very negative (-100) for 0.
    T_logits = torch.nn.Parameter(stochastic_to_real(T))
    f_logits = torch.nn.Parameter(stochastic_to_real(f))
    pfa.T_logits = T_logits
    pfa.f_logits = f_logits
    return pfa

def stochastic_to_real(input_tensor):
    zero_mask = (input_tensor == 0)
    one_mask = (input_tensor == 1)

    output_tensor = input_tensor.masked_fill(zero_mask, -100)
    output_tensor = output_tensor.masked_fill(one_mask, 100)

    return output_tensor


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
every = set_pfa_params(every, T, f)

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
some = set_pfa_params(some, T, f)


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
at_least_three = set_pfa_params(at_least_three, T, f)

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
at_least_four = set_pfa_params(at_least_four, T, f)

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
at_least_six_or_at_most_two = set_pfa_params(at_least_six_or_at_most_two, T, f)


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
at_most_three = set_pfa_params(at_most_three, T, f)


# check with shane
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
first_three = set_pfa_params(first_three, T, f)


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
}