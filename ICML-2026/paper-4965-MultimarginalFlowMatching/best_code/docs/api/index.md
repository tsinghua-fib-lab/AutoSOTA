# API reference

```{toctree}
:maxdepth: 1

otpfm
potentials
lambdas
networks
solvers
training
```

The top-level {mod}`otpfm` package re-exports the two pieces of the public API
used in [Quick start](../quickstart.md):

- {class}`otpfm.OTPFM` - the model.
- {class}`otpfm.Curriculum` - the OTP-FM $\alpha$ curriculum.

Everything else (potentials, lambda functions, networks, fixed-point solvers)
is exposed through submodules.
