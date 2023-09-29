"""Module defining the Probabilistic Automata used to learn semantic automata."""

import torch
from typing import Iterable

##############################################################################
# Helper functions
##############################################################################

def rand_init(shape: tuple[int], init_temperature: float) -> torch.Tensor:
    """Energy-based initialization from Normal distribution; higher init_temperature -> values are closer to 0."""
    return 1/init_temperature * torch.randn(shape)

def stochastic_to_real(input_tensor):
    zero_mask = (input_tensor == 0)
    one_mask = (input_tensor == 1)

    output_tensor = input_tensor.masked_fill(zero_mask, -100)
    output_tensor = output_tensor.masked_fill(one_mask, 100)

    return output_tensor

##############################################################################
# PFAModel class
##############################################################################

class PFAModel(torch.nn.Module):

    """A probabilistic finite-state automaton (PFA) is a tuple A = (Q, S, d, I, T, F, ) where

        - Q is a finite set of states
        - S is the alphabet
        - d subset Q x S x Q is a set of transitions
        - I: Q -> [0,1], the initial state probabilities
        - T: d -> [0,1], the state-transition probabilities
        - F: Q -> [0,1], the final state probabilities

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
        prob(i, q) = \sum_q prob(i-1, q') * T(q', x_i, q)
    """    

    def __init__(
        self,
        num_states: int,
        alphabet: Iterable[int],
        init_temperature: float = 1e-2,
        param_initial: bool = False,
        ) -> None:
        """Construct a PFA as a nn.Module.

        Args:
            num_states (int): the number of states of the pfa

            alphabet (Iterable): a set of symbols

            init_temperature (float): temperature parameter for random initialization

            param_initial (bool): whether to learn the initial state probabilities. Default is False, and the initial state will be fixed to the first state.
        """
        super(PFAModel, self).__init__()

        self.num_states = num_states
        self.alphabet = tuple(sorted(set(alphabet)))        
        self.num_symbols = len(self.alphabet)

        # Initial state params:
        # For each state, the probability of starting the sequence
        self.param_initial = param_initial
        if self.param_initial:
            self.i_logits = torch.nn.Parameter(rand_init([self.num_states], init_temperature))
        else:
            # Assume first state is always initial
            init = torch.zeros(num_states)
            init[0] = 1.
            self.init = init

        # Transition params:
        # The probability of transitioning to a state, given the current state and the current input string
        self.T_logits = torch.nn.Parameter(
            rand_init(
                [self.num_states, self.num_symbols, self.num_states], init_temperature)
            )

        # Final states params:
        # For each state, the probability that it is in the set of final states
        self.f_logits = torch.nn.Parameter(rand_init([self.num_states], init_temperature))

    @classmethod
    def from_probs(cls, transition: torch.Tensor, final: torch.Tensor, initial: torch.Tensor = None, alphabet: list[int] = [0,1]):
        """Takes stochastic parameters and initializes a PFAModel with the appropriate corresponding logits.
        
        Args:
            transition: a tensor of shape `[states, symbols, states]` representing the state transition probabilities

            final: a tensor of shape `[states]` representing the final probabilities

            initial: a tensor of shape `[states]` representing the initial probabilities. Default is None, and will be a 1-hot vector on the first state.            

            alphabet: a list of ints representing the symbols of the alphabet
        """

        if initial is None:
            initial = torch.zeros_like(final)
            initial[0] = 1.

        num_states = len(final)
        pfa = cls(
            num_states=num_states, 
            alphabet=alphabet, 
            init_temperature=1e-2,
            param_initial=True,            
        )
        # softmax is not invertible, but since we're creating delta functions, we can just choose very large (100) for 1, and very negative (-100) for 0.
        i_logits = torch.nn.Parameter(stochastic_to_real(initial))
        T_logits = torch.nn.Parameter(stochastic_to_real(transition))
        f_logits = torch.nn.Parameter(stochastic_to_real(final))

        pfa.i_logits = i_logits
        pfa.T_logits = T_logits
        pfa.f_logits = f_logits

        return pfa


    def forward_algorithm(self, sequence: torch.Tensor) -> torch.Tensor:
        """Compute the (log) probability of a sequence under the PFA using the Forward algorithm (Vidal et al., 2005a, Section 3).
        """
        num_transitions = len(sequence) + 1

        # shape `[num_transitions, num_states]`
        log_alpha = torch.log(torch.zeros(num_transitions, self.num_states, requires_grad=True, device=next(self.parameters()).device))

        # Obtain the forward probabilities for the first position
        if self.param_initial:
            log_alpha[0] = torch.nn.functional.logsigmoid(self.i_logits)
        else:
            log_alpha[0] = torch.log(self.init)

        # Create a tensor for sequence symbols
        symbol_indices = torch.tensor([self.symbol_to_index(symbol) for symbol in sequence], dtype=torch.long)

        # # Don't know why the following vectorization won't work.
        # for i in range(1, num_transitions):
        #     transition_probs = log_alpha[i-1] + self.transition(symbol_indices[i-1])
        #     log_alpha[i] = torch.logsumexp(transition_probs, -1)

        # Iterate over sequence 
        for i in range(1, num_transitions):
            for state_idx in range(self.num_states):
                # T(q', a, q)
                transition_probs = torch.nn.functional.log_softmax(self.T_logits, -1)[:, symbol_indices[i-1], state_idx]

                log_alpha[i, state_idx] = torch.logsumexp(
                    log_alpha[i-1] + transition_probs, -1,
                )

        # Compute total prob the automaton ends in final state
        total_log_alpha = torch.logsumexp(
            # prob(|x|, q) * F(q)
            log_alpha[-1] + torch.nn.functional.logsigmoid(self.f_logits), -1
        )
        return total_log_alpha


    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """The forward pass through the computation graph for the PFA.

        Args:
            x: Tensor of Ints of shape `[batch_size, sequence_length]` containing the input sequences

            sequence_lengths: Tensor of Ints of shape `[batch_size,]` containing the original sequence lengths of each example in batch, before max padding.

        Returns:
            out: Tensor of output probabilities of shape `[batch_size,]`.
        """
        # Create a list of model outputs
        log_probs = torch.stack([
            self.forward_algorithm(tuple(seq[:lengths[idx]].tolist()))
            for idx, seq in enumerate(x)
        ])
        # NOTE: I've observed that all values of probs are always basically within 0,1, but slightly numerically unstable. So we lose little by clamping.
        probs = log_probs.exp()
        out = torch.clamp(probs, 0, 1)
        return out
    
    def prob_sequence(self, seq: tuple[int]) -> float:
        """Probability of a binary sequence; helper function for testing since we mostly use forward for learning."""
        return self.forward_algorithm(seq).exp()

    
    def symbol_to_index(self, symbol: int):
        return self.alphabet.index(symbol)    
