# easy_meanflow (vendored subset)

Verbatim copy of the `dnnlib/`, `torch_utils/` and `training/` modules from the
authors' `easy_meanflow` repo (HEAD `109a21e`), which trains the one-step
**mean-flow** model used by the fast black-hole reconstruction
(`--posterior meanflow`).

Only these modules are vendored — they are what
`algo.meanflow_posterior.load_meanflow_net` needs to **unpickle** a mean-flow
checkpoint (the checkpoint stores `ema` / `loss_fn` / `augment_pipe` objects via
StyleGAN3-style `torch_utils.persistence`, so `dnnlib`, `torch_utils` and
`training.networks_mf` must be importable). Training scripts, datasets and the
generation CLI are not included.

The checkpoint itself is not public yet; see
[`experiments/black_hole/download.sh`](../../experiments/black_hole/download.sh).
