import torch
from qal.quantifier import quantifier_map
from qal.data_gen import generate_up_to_length, string_to_int_tuple

class TestQuantifierPFA:

    max_length = 10
    strings = generate_up_to_length(max_length)

    def test_every(self):
        # Test the qpfa and the python 'all' accept the same strings
        qpfa = quantifier_map["every"]

        for string in TestQuantifierPFA.strings:
            tup = string_to_int_tuple(string)
            if all(tup):
                assert qpfa.forward_algorithm(tup) == 0.
            else:
                if not torch.isneginf(qpfa.forward_algorithm(tup)):
                    breakpoint()

    def test_some(self):
        # Test the qpfa and the python 'any' accept the same strings
        qpfa = quantifier_map["some"]

        for string in TestQuantifierPFA.strings:
            tup = string_to_int_tuple(string)
            if any(tup):
                assert qpfa.forward_algorithm(tup) == 0.
            else:
                if not torch.isneginf(qpfa.forward_algorithm(tup)):
                    breakpoint()

    def test_at_least_three(self):
        # Test the qpfa and the python 'any' accept the same strings
        qpfa = quantifier_map["at_least_three"]

        for string in TestQuantifierPFA.strings:
            tup = string_to_int_tuple(string)
            if sum(tup) >= 3:
                assert qpfa.forward_algorithm(tup) == 0.
            else:
               if not torch.isneginf(qpfa.forward_algorithm(tup)):
                    breakpoint()
