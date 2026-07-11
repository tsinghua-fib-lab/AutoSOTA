# Comparison protocol

SIGS returns symbolic closed-form candidates rather than only a numerical surrogate.
For fair comparisons with symbolic regression, neural PDE solvers, PINNs, computer
algebra systems, or LLM-based equation tools, report both solution quality and
representation quality.

## Minimum metrics

For each problem, report:

- PDE residual on the stated evaluation grid;
- boundary and/or initial condition error;
- relative L2 error when a known or trusted reference solution exists;
- wall-clock time and hardware;
- random seed or search budget;
- whether the method returns an explicit symbolic expression;
- expression complexity, such as number of atoms/nodes or production rules;
- whether the candidate satisfies constraints exactly or approximately.

## Suggested table columns

| Method | Output type | Residual | BC/IC error | Relative L2 | Time | Symbolic complexity | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| SIGS | Closed-form expression | | | | | | |
| PINN / neural solver | Numerical surrogate | | | | | N/A | |
| Symbolic regression baseline | Expression | | | | | | |
| Computer algebra / LLM baseline | Expression or failure | | | | | | |

## Important distinctions

- Do not compare only grid error when one method outputs an interpretable symbolic expression and another outputs a numerical field.
- State whether problem-specific dictionaries, ansatzes or human-provided basis functions were used.
- State whether constants were refined after structural discovery.
- For stochastic search methods, report multiple seeds or a fixed search budget.
- For unknown closed-form problems, compare against a trusted numerical reference and inspect boundary/initial condition satisfaction.
