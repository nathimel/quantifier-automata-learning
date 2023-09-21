
from qal.quantifier import quantifier_map
from qal.data_gen import generate_up_to_length, string_to_int_tuple

class TestQuantifierPFA:

    def test_every(self):
        # Test the qpfa and the python 'all' accept the same strings
        qpfa = quantifier_map["every"]
        max_length = 10
        strings = generate_up_to_length(max_length)

        for string in strings:
            tup = string_to_int_tuple(string)
            if all(tup):
                assert qpfa.logp_string(tup) == 0.


    def test_some(self):
        # Test the qpfa and the python 'any' accept the same strings
        qpfa = quantifier_map["some"]
        max_length = 10
        strings = generate_up_to_length(max_length)

        for string in strings:
            tup = string_to_int_tuple(string)
            if any(tup):
                assert qpfa.logp_string(tup) == 0.