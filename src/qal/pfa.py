"""Module defining the Probabilistic Automata used to learn semantic automata."""

import torch
from typing import Iterable

def rand_init(shape: tuple[int], init_temperature: float) -> torch.Tensor:
    """Energy-based initialization from Normal distribution; higher init_temperature -> values are closer to 0."""
    return 1/init_temperature * torch.randn(shape)


class PFAModel(torch.nn.Module):

    """A probabilistic finite-state automaton (PFA) is a tuple A = (Q, S, d, I, T, F, ) where

        - Q is a finite set of states
        - S is the alphabet
        - d subset Q x S x Q is a set of transitions
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
        alphabet: Iterable[int],
        init_temperature: float = 1e-2,
        ) -> None:
        """Construct a PFA as a nn.Module.

        Args:
            num_states: the number of states of the pfa

            alphabet: a set of symbols

            T: a Tensor of shape `[num_states, len(alphabet), num_states]`, the state-transition probabilities

            final: a Tensor of shape `[num_states,]`, the final state probabilities
        """
        super(PFAModel, self).__init__()

        self.num_states = num_states
        self.alphabet = tuple(sorted(set(alphabet)))        
        self.num_symbols = len(self.alphabet)

        # Transition params:
        # The probability of transitioning to a state, given the current state and the current input string
        self.T_logits = torch.nn.Parameter(
            rand_init(
                [self.num_states, self.num_symbols, self.num_states], init_temperature)
            )

        # Final states params:
        # For each state, the probability that it is in the set of final states
        self.f_logits = torch.nn.Parameter(rand_init([self.num_states], init_temperature))

        # Assume first state is always initial
        init = torch.zeros(num_states)
        init[0] = 1.
        self.init = init

    def forward_algorithm(self, sequence: torch.Tensor) -> torch.Tensor:
        """Compute the (log) probability of a sequence under the PFA using the Forward algorithm (Vidal et al., 2005a, Section 3).
        """

        num_transitions = len(sequence) + 1
        log_alpha = torch.log(torch.zeros(num_transitions, self.num_states, requires_grad=True))

        # Initialize the forward probabilities for the first position
        log_alpha[0] = torch.log(self.init)

        # Create a tensor for sequence symbols
        symbol_indices = torch.tensor([self.symbol_to_index(symbol) for symbol in sequence], dtype=torch.long)

        # # Don't know why the following won't work.
        # for i in range(1, num_transitions):
        #     transition_probs = log_alpha[i-1] + self.transition(symbol_indices[i-1])
        #     log_alpha[i] = torch.logsumexp(transition_probs, -1)

        # Iterate over sequence 
        for i in range(1, num_transitions):
            for state_idx in range(self.num_states):
                # p(q', a, q)
                transition_probs = torch.nn.functional.log_softmax(self.T_logits, -1)[:, symbol_indices[i-1], state_idx]

                log_alpha[i, state_idx] = torch.logsumexp(
                    log_alpha[i-1] + transition_probs, -1,
                )

        # Compute sum_q prob(|x|, q) * F(q)
        total_log_alpha = torch.logsumexp(log_alpha[-1] + self.f_logits, -1)
        return total_log_alpha


    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """The forward pass through the computation graph for the PFA.

        Args:
            x: Tensor of Ints of shape `[batch_size, sequence_length]` containing the input sequences

            sequence_lengths: Tensor of Ints of shape `[batch_size,]` containing the original sequence lengths of each example in batch, before max padding.

        Returns:
            out: Tensor of output logits of shape `[batch_size,]`.
        """
        # Create a list of model outputs
        return torch.stack([
            self.forward_algorithm(tuple(seq[:lengths[idx]].tolist()))
            for idx, seq in enumerate(x)
        ])

    
    def symbol_to_index(self, symbol: int):
        return self.alphabet.index(symbol)    
