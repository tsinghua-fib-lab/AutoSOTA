# Final independent-head method

For fold f, collect its own Stage 3/4/5 spatial statistics in the order
stage3 mean/max/population-std, stage4 mean/max/population-std,
stage5 mean/max/population-std. Channel counts are 256/320/320, giving 768/960/960 features.

For each stage: `h = GELU(Linear16(LayerNorm(x)))` and
`r = B * tanh(Linear3(h) / B)`. All output weights and biases start at zero.
The corrected logits are `exp(t) * (released_logits + bias) + r3 + r4 + r5`.
Fold 0 has no bias; folds 0/2/6 fix temperature to 1; folds 3/4 bound it to [0.5,1.5].
Folds 1/5 use only `released_logits + bias`, clipped to ±0.2.

Residual heads optimize CE plus `lambda * KL(softmax(z_released / 2) || softmax(z_new / 2))`.
The historical implementation does not multiply this KL by temperature squared.
Bias-only heads use CE without KL. AdamW uses 3e-4 and weight decay 1e-3;
residual-head LR decays by cosine over exactly 3,000 optimizer steps. Bias-only LR stays constant.

Heads are initialized with local, fold-specific random generators. Elementwise
linear maps preserve the numerical procedure used by the experiment. Fold training
seeds are 3107..3113. Checkpoint selection: maximum validation Challenge score,
then lower validation NLL on ties within 1e-12; epoch-zero identity is a candidate.

The final prediction is a fixed arithmetic mean of seven softmax probability vectors.
No fold weighting, fold removal, cross-fold input or parameter sharing is allowed.

The outer search first maximized the number of improved public-test fold scores, then
ensemble score when all seven improved. This outer process used test feedback; it is
not an independent test protocol. The public release fixes the already-selected v29
method rather than re-running an automatic search or requiring LLM/API credentials.
