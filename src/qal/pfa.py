"""Module defining the Probabilistic Automata used to learn semantic automata."""

import torch
from typing import Iterable

class PFAModel(torch.nn.Module):

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

        # Transition params
        self.transition = TransitionModel(self.num_states, self.num_symbols, init_temperature)

        # Final state distribution params
        self.final = FinalModel(self.num_states, init_temperature)

        # Assume first state is always initial
        init = torch.zeros(num_states)
        init[0] = 1.
        self.init = init

    def forward_algorithm(self, sequence: torch.Tensor) -> torch.Tensor:
        """Compute the probability of a sequence under the PFA using the Forward algorithm (Vidal et al., 2005a, Section 3).
        """

        num_transitions = len(sequence) + 1
        log_alpha = torch.log(torch.zeros(num_transitions, self.num_states))

        # Initialize the forward probabilities for the first position
        log_alpha[0] = torch.log(self.init)
        
        # Iterate over sequence 
        for i in range(1, num_transitions):
            
            # Although we have already 'transitioned' into the initial state, we have not read the first symbol yet.
            symbol_idx = self.symbol_to_index(sequence[i-1])

            for state_idx in range(self.num_states):

                log_alpha[i, state_idx] = torch.logsumexp(torch.tensor([
                    # p(i-1, q')
                    log_alpha[i-1, from_state_idx]
                    # p(q', a, q)
                    + self.transition(from_state_idx, symbol_idx, state_idx)
                    for from_state_idx in range(self.num_states)
                    ]), -1)
        
        # Compute the total probability by summing over the final state probabilities
        total_log_alpha = torch.logsumexp(torch.tensor([
            # sum_q prob(|x|, q) * F(q)
            log_alpha[-1, state_idx] + self.final(state_idx)
            for state_idx in range(self.num_states)
        ]), -1)

        return total_log_alpha


    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        """The forward pass through the computation graph for the PFA.

        # NOTE: for now we're not vectorizing/parallelizing for simplicity, but that is desirable.

        Args:
            x: Tensor of Ints of shape `[batch_size, sequence_length]` containing the input sequences

            sequence_lengths: Tensor of Ints of shape `[batch_size,]` containing the original sequence lengths of each example in batch, before max padding.

        Returns:
            out: Tensor of output logits of shape `[batch_size,]`.
        """
        return torch.tensor([self.forward_algorithm(
                tuple(seq[:lengths[idx]].tolist())
                ) 
            for idx, seq in enumerate(x)
            ])
    
    def symbol_to_index(self, symbol: int):
        return self.alphabet.index(symbol)    


# (sub)Modules for the PFA parameters.
# Note the log_softmax in each forward pass, as we are computing log probabilities.
class TransitionModel(torch.nn.Module):

    def __init__(self, num_states, num_symbols, init_temperature) -> None:
        super(TransitionModel, self).__init__()
        self.T_logits = torch.nn.Parameter(rand_init([num_states, num_symbols, num_states], init_temperature))

    def forward(self, from_state_idx, symbol_idx, state_idx) -> torch.Tensor:
        """Forward pass for p(q', a, q), the probability of transitioning to q from q' and accepting symbol a."""
        log_T = torch.nn.functional.log_softmax(self.T_logits, dim=-1)
        return log_T[from_state_idx, symbol_idx, state_idx]

class FinalModel(torch.nn.Module):

    def __init__(self, num_states, init_temperature) -> None:
        super(FinalModel, self).__init__()
        self.f_logits = torch.nn.Parameter(rand_init([num_states], init_temperature))

    def forward(self, state_idx: int) -> torch.Tensor:
        """Forward pass for the final state distribution logits.
        """
        log_f = torch.nn.functional.log_softmax(self.f_logits)
        return log_f[state_idx]
    


def rand_init(shape: tuple[int], init_temperature: float) -> torch.Tensor:
    """Energy-based initialization from Normal distribution; higher init_temperature -> values are closer to 0."""
    return 1/init_temperature * torch.randn(shape)




class PFA:

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
        self.alphabet = tuple(sorted(set(alphabet)))
        self.init = init
        self.T = T
        self.final = final

    def logp_string(self, x: tuple[int]) -> float:
        """Compute the log probability of the string x using the Forward algorithm."""

        # Forward algorithm
        num_transitions = len(x) + 1
        logprob = torch.log(torch.zeros(num_transitions, self.num_states))

        # Initialize the forward probabilities for the first position
        logprob[0] = torch.log(self.init)
        
        # Iterate over sequence 
        for i in range(1, num_transitions):
            
            # Although we have already 'transitioned' into the initial state, we have not read the first symbol yet.
            symbol_idx = self.symbol_to_index(x[i-1])

            for state_idx in range(self.num_states):
                logprob[i, state_idx] = torch.logsumexp(torch.tensor([
                    # p(i-1, q')
                    logprob[i-1, from_state_idx]
                    # p(q', a, q)
                    + torch.log(self.T[from_state_idx, symbol_idx, state_idx])
                    for from_state_idx in range(self.num_states)
                    ]), -1)
        
        # Compute the total probability by summing over the final state probabilities
        total_logprob = torch.logsumexp(torch.tensor([
            # sum_q prob(|x|, q) * F(q)
            logprob[-1, state_idx] + torch.log(self.final[state_idx])
            for state_idx in range(self.num_states)
        ]), -1)

        return total_logprob
    
    def prob(self, seqs: torch.Tensor, seq_lengths: torch.Tensor) -> torch.Tensor: 
        """Convenience function for getting a tensor of probabilities for a batch of strings.
        
        Args:
            seqs: a Tensor of ints, corresponding to binary strings padded by 2s to be of consistent length

            seq_lengths: a Tensor of ints, corresponding to the original sequence lengths of binary strings before padding
        """
        log_probs = torch.tensor([
            self.logp_string(
                tuple(seq[:seq_lengths[idx]].tolist())
                ) 
            for idx, seq in enumerate(seqs)
            ])
        probs = log_probs.exp()
        return probs


    def symbol_to_index(self, symbol: int):
        return self.alphabet.index(symbol)