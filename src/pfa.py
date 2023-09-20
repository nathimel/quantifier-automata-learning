"""Module defining the Probabilistic Automata used to learn semantic automata."""

import torch

class PFA:

    """A probabilistic finite-state automaton (PFA) is a tuple A = (Q, S, d, I, T, F, ) where

        - Q is a finite set of states
        - S is the alphabet
        - d \subset Q x S x Q is a set of transitions
        - I: Q -> [0,1] is the initial state probability distribution
        - T: d -> [0,1] is the state-transition probability distribution
        - F: Q -> [0,1] is the final state probability distribution

    Given a PFA A, the process of generating (accepting) a string proceeds as follows:

        Initialization: choose (w.r.t. I) one state q_0 in Q as the initial state. Define q_0 as the current state.

        Generation: Let q be the current state. Decide whether to stop with probability F(q), or to make a move (q, s, q') with probability T(q, s, q'), where s is a symbol \in S and q' is a state \in Q. Output (input) and set the current state to q'.

    The probability of accepting a (finite) string x \in S* is:

        prob(x) = \sum_q prob(|x|, q) * F(q)
    
    where prob(i, q) the probability of parsing the prefix (x_1, ..., x_i) and reaching state q, for 0 <= i <= |x| is defined:

        prob(i, q) = 
        \sum_{pi \in \Pi} I(pi_0) * \Prod_j^i T(pi_{j-1}, x_j, s_j) * \1(q, s_i)

    where \Pi is the set of all paths (transition sequences) leading to the acceptance of x. This probability, of parsing an i-substring of x and reaching state q, can be computed using the recursive Forward algorithm:

        prob(0, q) = I(q)
        prob(i, q) = \sum_q prob(i-1, q') * P(q', x_i, q)
    """

    def __init__(
        self,
        num_states: int,
        alphabet: set[int],
        init: torch.Tensor,
        T: torch.Tensor,
        final: torch.Tensor,
        ) -> None:
        """Construct a pfa.

        Args:
            num_states: the number of states of the pfa

            alphabet: a set of symbols

            init: a Tensor of shape `[num_states,]` the initial state probabilities

            T: a Tensor of shape `[num_states, len(alphabet), num_states]`, the state-transition probabilities

            final: a Tensor of shape `[num_states,]`, the final state probabilities
        """

        self.num_states = num_states
        self.alphabet = tuple(sorted(alphabet))
        self.init = init
        self.T = T
        self.final = final

    def logp_string(self, x: tuple[int]) -> float:
        """Compute the log probability of the string x using the Forward algorithm."""

        # Forward algorithm
        logprob = torch.zeros(len(x), self.num_states)

        # iterate over all initial states
        for state_idx in range(self.num_states):

            # first element of sequence
            logprob[0, state_idx] = torch.log(self.init[state_idx])

            # iterate over sequence
            for i in range(1, len(x)):
                symbol_idx = self.symbol_to_index(x[i])
                logprob[i, state_idx] = sum([
                    logprob[i-1, from_state_idx, state_idx] # p(i-1, q')
                    + self.T[from_state_idx, symbol_idx, state_idx] # p(q',a,q)
                    for from_state_idx in range(self.num_states)
                ])
        
        # compute total probability
        total_logprob = sum([
            # prob(|x|, q) * F(q)
            logprob[-1, state_idx] + self.final[state_idx] 
            for state_idx in range(self.num_states)
        ])

        return total_logprob

    def symbol_to_index(self, symbol: int):
        return self.alphabet.index(symbol)