# infer/

Sequence generation and sampling under different policy parameterizations (safe, unconstrained, CPC-constrained).

## Key files

- **rejection_sampling.py** — `generate_proposals_for_AR_sampling()`: generates proposals from a specified policy, computes likelihoods under all models, constrains likelihoods per CPC bounds, and performs acceptance-rejection or independent Metropolis-Hastings sampling. In direct mode, uses in-memory functions to bypass subprocess spawning and file I/O. `accept_reject_sample_and_get_likelihoods()`: wrapper that handles the full AR pipeline including likelihood computation; pre-loads the generation ModelClient across AR iterations in direct mode.
- **iterative_generation2.py** — `run_iterative_generation_inmemory()`: core generation loop that accepts a pre-loaded ModelClient and returns a DataFrame directly. `run_iterative_generation()`: file-I/O wrapper that loads data/model, calls the in-memory function, and writes results to disk.
- **generation_utils.py** — `get_temperatures()`: adaptively scales generation temperature based on previous iteration's Hamming distance to ensure sufficient exploration.

## Sampling algorithms

The rejection sampling supports two modes:
- **Accept-reject**: accepts proposal x with probability min(1, pi_constrained(x) / (M * pi_proposal(x)))
- **Independent Metropolis-Hastings**: keeps previous sample if proposal is rejected, with acceptance ratio based on target/proposal likelihood ratios
