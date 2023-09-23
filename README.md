# quantifier-automata-learning

This codebase represents a replication of the [experiments](https://github.com/shanest/quantifier-rnn-learning) of Steinert-Threlkeld and Szymanik 2018, "Learnability and Semantic Universals". Instead of testing the learnability of quantifier universals by LSTMs, we train simple probabilistic finite-state automata (PFAs) -- still using gradient descent -- to approximate semantic automata.

## Requirements

Create the conda environment:

- Get the required packages by running

    `conda env create -f environment.yml`

## Replicating experimental results

Run the following commands to replicate the 3 individual experiments.

### Monotonicity

Experiment 1a,b

`python src/main.py learning.patience=1000 learning.learning_rate=1e-4 quantifier=at_least_six_or_at_most_two learning.num_states=7 learning.epochs=4000`

`python src/main.py learning.patience=1000 learning.learning_rate=1e-4 quantifier=at_least_four learning.num_states=5 learning.epochs=4000`

`python src/main.py learning.patience=1000 learning.learning_rate=1e-4 quantifier=at_most_three learning.num_states=5 learning.epochs=4000`

### Quantity

Experiment 2a

`python src/main.py learning.patience=1000 learning.learning_rate=1e-4 quantifier=at_least_three learning.num_states=4 learning.epochs=4000`

`python src/main.py learning.patience=1000 learning.learning_rate=1e-4 quantifier=first_three learning.num_states=5 learning.epochs=4000`

## Hydra

This codebase uses [hydra](https://hydra.cc), so you can easily sweep over parameters using `--multirun`. See the [conf](conf/) folder for overrides.
