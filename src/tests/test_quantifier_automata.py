
import torch
from typing import Callable
from qal.pfa import PFAModel
from qal.quantifier import quantifier_map
from qal.data_gen import generate_up_to_length, string_to_int_tuple

def check_logprobs(
        qpfa: PFAModel, 
        strings: list[str], 
        func: Callable[[tuple[int]],int]):
    for string in strings:
        tup = string_to_int_tuple(string)
        if func(tup):
            assert qpfa.forward_algorithm(tup) == 0.
            # if qpfa.forward_algorithm(tup) == 0:
                # breakpoint()
        else:
            assert torch.isneginf(qpfa.forward_algorithm(tup))
            # if torch.isneginf(qpfa.forward_algorithm(tup)):
            #     breakpoint()

class TestQuantifierPFA:

    def test_every(self):
        # Test the qpfa and the python 'all' accept the same strings
        qpfa = quantifier_map["every"]
        max_length = 10
        strings = generate_up_to_length(max_length)
        check_logprobs(qpfa, strings, all)

    def test_some(self):
        # Test the qpfa and the python 'any' accept the same strings
        qpfa = quantifier_map["some"]
        max_length = 10
        strings = generate_up_to_length(max_length)
        check_logprobs(qpfa, strings, any)

    def test_at_least_three(self):
        qpfa = quantifier_map["at_least_three"]
        strings = generate_up_to_length(10)
        check_logprobs(qpfa, strings, lambda x: sum(x) >= 3)

    def test_at_least_four(self):
        qpfa = quantifier_map["at_least_four"]
        strings = generate_up_to_length(10)
        check_logprobs(qpfa, strings, lambda x: sum(x) >= 4)

    def test_at_least_six_or_at_most_two(self):
        qpfa = quantifier_map["at_least_six_or_at_most_two"]
        strings = generate_up_to_length(10)
        check_logprobs(qpfa, strings, lambda x: sum(x) >= 6 or sum(x) <= 2)
