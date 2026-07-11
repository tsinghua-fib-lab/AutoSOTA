# SIGS: A Neuro-Symbolic Solver for Closed-Form Differential Equation Solutions

Analytical solutions of differential equations are unusually valuable: they are exact, interpretable, reusable across parameter regimes, and often reveal structure that a numerical approximation hides. The difficulty is that discovering such solutions usually requires expert intuition, a carefully chosen Ansatz, or a search over an enormous symbolic space. SIGS, short for Symbolic Iterative Grammar Solver, was designed to automate part of this process by combining formal grammars, latent representation learning, and physics-based residual minimisation.

## The problem: symbolic solution discovery is combinatorial

Many scientific machine learning methods approximate the solution of a differential equation with a neural network or a numerical surrogate. These approaches can be extremely useful, but they usually return a function approximator rather than a closed-form expression. Symbolic regression goes in the opposite direction: it searches for an explicit mathematical expression. However, for differential equations, the search space becomes combinatorial very quickly. Expressions can contain arithmetic operations, trigonometric functions, exponentials, products, compositions, constants, and multiple variables. Even before checking whether a candidate satisfies a PDE, the number of possible expression trees can be enormous.

SIGS addresses this by making the symbolic search structured from the beginning. Instead of exploring arbitrary strings, it uses a context-free grammar to generate syntactically valid mathematical expressions. This grammar defines the symbolic language in which candidate solution blocks are allowed to live.

## The core idea: search in a grammar-induced latent space

SIGS has two stages. In the first stage, a Grammar Variational Autoencoder embeds grammar-generated expressions into a continuous latent space. This turns a discrete symbolic search problem into a more organised search over a learned manifold of mathematical expressions. The method samples candidate structures from this latent space, decodes them into symbolic expressions, and evaluates them using the differential equation residual together with boundary and initial conditions.

In the second stage, SIGS freezes the discovered symbolic structure and exposes only its numerical constants as trainable parameters. These constants are then refined with JAX automatic differentiation by directly minimising the physics-based residual. In this way, SIGS separates structural discovery from numerical parameter optimisation.

This division is important. The hard symbolic problem is handled by grammar-guided latent search, while the continuous optimisation problem is handled by differentiable residual minimisation.

## What makes SIGS different?

SIGS is not a neural surrogate solver in the usual sense. It does not only produce pointwise predictions on a grid. Its output is an explicit mathematical expression. That makes the result interpretable, portable, and easy to evaluate once discovered.

SIGS is also not ordinary symbolic regression. It does not search blindly over arbitrary expression trees. The grammar constrains candidate solutions to valid mathematical forms, while the learned latent space gives the search a geometry. Similar expressions are represented in related regions of the latent space, which makes the search less purely combinatorial.

Finally, SIGS is equation-driven rather than data-driven. Candidate expressions are scored by how well they satisfy the differential equation and its conditions, not by fitting a dataset of solution values.

## Examples and results

SIGS was validated on several differential equation systems, including Burgers, diffusion, advection, damped wave, KdV, Poisson with a Gaussian source, and coupled nonlinear PDE systems such as shallow water equations. On classical benchmarks with known closed-form solutions, SIGS recovers machine-precision analytical expressions. On Poisson--Gauss, where no standard closed-form solution is available, it produces an accurate symbolic approximation. On grammar misspecification tests, it can recover an equivalent solution form even when the original atom is removed from the grammar.

The broader point is not only that SIGS solves a set of benchmark equations. It shows that grammar-guided neuro-symbolic search can be used as a practical route toward analytical solution discovery.

## When should you cite SIGS?

SIGS is most relevant if your work concerns closed-form or analytical solution discovery for ODEs and PDEs, neuro-symbolic scientific machine learning, grammar-guided symbolic search, PDE residual-based symbolic optimisation, Grammar-VAE or latent-space search over mathematical expressions, or data-free symbolic PDE solving.

## Links

- Paper: https://arxiv.org/abs/2502.01476
- ICML 2026 poster: https://icml.cc/virtual/2026/poster/63043
- Code: https://github.com/oroikono/SIGS
- Project page: https://oroikono.github.io/sigs-paper-site/
- 5-minute video: https://www.youtube.com/watch?v=a9MMvKVGhuQ

## Citation

```bibtex
@misc{oikonomou2026neurosymbolic,
  title         = {Neuro-Symbolic {AI} for Analytical Solutions of Differential Equations},
  author        = {Oikonomou, Orestis and Lingsch, Levi and Grund, Dana and Mishra, Siddhartha and Kissas, Georgios},
  year          = {2026},
  eprint        = {2502.01476},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi           = {10.48550/arXiv.2502.01476},
  url           = {https://arxiv.org/abs/2502.01476}
}
```
