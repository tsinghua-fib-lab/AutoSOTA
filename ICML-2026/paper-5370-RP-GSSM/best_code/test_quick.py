"""Quick integration test."""
import jax
import jax.numpy as np
import jax.random as jr
import optax

print("JAX devices:", jax.devices())

from rp_ssm import datasets, utils, dists, recognition, distmaps, rpm, config, training
from rp_ssm import datasets_utils

# Generate small linear dataset for testing
key = jr.PRNGKey(0)
small_data = datasets_utils.generate_linear_data(1, 4, 8, 32, 50, 0.1, key)
print("Small data shapes:")
print("  train_obs:", small_data.train_obs[0].shape)
print("  train_states:", small_data.train_states.shape)

# Build model with small latent dim
latent_dim = 4
prior = dists.LGStationaryParam(
    start_from_invariant=True,
    stay_at_invariant=True,
    opt_params=["A"],
    A=0.5 * np.eye(latent_dim),
)

network = recognition.MLP([32, 32])
rec = [
    recognition.RPMRecognition(
        network=network,
        dist_map=distmaps.MVNCholesky(latent_dim),
        constant_cov=True,
    )
]

model = rpm.RPSSM(prior=prior, recognition=rec)

cfg = config.Config(
    num_iter=50,
    prior_lr=1e-3,
    rec_lr=(1e-3,),
    batch_size=32,
    jit=True,
    stabilize_A="clip",
    seed=0,
)

free_energy = rpm.ConstrainedIVFreeEnergy(model=model)

trainer = training.Trainer(free_energy=free_energy, config=cfg)
print("Starting quick training test...")
trainer.fit(small_data.train_data, use_pbar=False)
print("Training test passed! Final loss:", trainer.loss_tot[-1])

_, posterior = trainer.apply(small_data.val_data)
means = posterior.params["means"]
r2, _ = utils.linear_r2(means, small_data.val_states)
print(f"Val Linear R2: {r2:.4f}")
print("All systems go!")
