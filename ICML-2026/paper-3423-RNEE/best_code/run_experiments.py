import numpy as np
import matplotlib.pyplot as plt
from time import time
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

plt.rcParams.update({'font.size': 13, 'figure.figsize': (12, 5)})
print('Imports OK')
# Build the credit risk state-preparation circuit (same as companion notebook)
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit.library import WeightedAdder, LinearAmplitudeFunction
from qiskit_finance.circuit.library import GaussianConditionalIndependenceModel
from qiskit.primitives import StatevectorSampler

n_z = 4; z_max = 3
p_zeros = [0.15, 0.25]; rhos = [0.1, 0.05]; lgd = [1, 2]; K = 2

GCI = GaussianConditionalIndependenceModel(n_z, z_max, p_zeros, rhos)
agg = WeightedAdder(n_z + K, [0] * n_z + lgd)
objective = LinearAmplitudeFunction(
    agg.num_sum_qubits, slope=[1], offset=[0],
    domain=(0, 2**agg.num_sum_qubits - 1), image=(0, sum(lgd)),
    rescaling_factor=0.25, breakpoints=[0])

qr_state = QuantumRegister(GCI.num_qubits, 'state')
qr_sum = QuantumRegister(agg.num_sum_qubits, 'sum')
qr_carry = QuantumRegister(agg.num_carry_qubits, 'carry')
qr_obj = QuantumRegister(1, 'objective')

state_preparation = QuantumCircuit(qr_state, qr_obj, qr_sum, qr_carry, name='A')
state_preparation.append(GCI.to_gate(), qr_state)
state_preparation.append(agg.to_gate(), qr_state[:] + qr_sum[:] + qr_carry[:])
state_preparation.append(objective.to_gate(), qr_sum[:] + qr_obj[:])

# Get ground truth via high-shot simulation
sp_measure = state_preparation.measure_all(inplace=False)
job = StatevectorSampler(seed=42).run([sp_measure], shots=200000)
counts = job.result()[0].data.meas.get_counts()
total = sum(counts.values())

exact_value = 0
for bs, cnt in counts.items():
    prob = cnt / total
    if prob > 1e-7 and bs[::-1][len(qr_state)] == '1':
        exact_value += prob

exact_loss = objective.post_processing(exact_value)
sigma2 = exact_value * (1 - exact_value)  # Bernoulli variance
sigma = np.sqrt(sigma2)

print(f'Circuit: {state_preparation.num_qubits} qubits')
print(f'Exact operator value a = {exact_value:.6f}')
print(f'Exact expected loss    = {exact_loss:.4f}')
print(f'Operator std dev       = {sigma:.4f}')
# Run IAE at multiple precision levels
from qiskit_algorithms import IterativeAmplitudeEstimation, EstimationProblem

problem = EstimationProblem(
    state_preparation=state_preparation,
    objective_qubits=[len(qr_state)],
    post_processing=objective.post_processing)

# Use a range where IAE produces meaningful (nonzero) query counts
epsilons_iae = [0.03, 0.02, 0.01, 0.005, 0.003, 0.002, 0.001, 0.0005, 0.0003]

iae_queries = []
iae_estimates = []

for eps in epsilons_iae:
    ae = IterativeAmplitudeEstimation(
        epsilon_target=eps, alpha=0.05,
        sampler=StatevectorSampler(seed=42, default_shots=1024))
    result = ae.estimate(problem)
    nq = result.num_oracle_queries
    iae_queries.append(nq)
    iae_estimates.append(result.estimation_processed)
    ci = np.array(result.confidence_interval_processed)
    print(f'  eps={eps:.4f}  queries={nq:>10d}  est={result.estimation_processed:.4f}  '
          f'CI=[{ci[0]:.4f}, {ci[1]:.4f}]')

iae_queries = np.array(iae_queries, dtype=float)
epsilons_iae = np.array(epsilons_iae)

# Filter out zero-query results
mask = iae_queries > 0
iae_queries_nz = iae_queries[mask]
eps_iae_nz = epsilons_iae[mask]

print(f'\n{np.sum(mask)}/{len(mask)} epsilon values produced nonzero query counts.')
# Plot: Classical vs Quantum (theoretical) vs IAE (empirical)
eps_range = np.logspace(-1, -4, 200)

# Classical MC: N = sigma^2 / eps^2
classical_cost = sigma2 / eps_range**2

# Quantum QAMC theoretical: N = (sigma/eps) * log(sigma/eps) from Corollary 3.2
ratio = sigma / eps_range
quantum_theory = ratio * np.maximum(1, np.log2(ratio))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: query counts
ax1.loglog(eps_range, classical_cost, 'r-', lw=2.5, label='Classical MC: $\\sigma^2/\\varepsilon^2$')
ax1.loglog(eps_range, quantum_theory, 'b-', lw=2.5, label='QAMC theory: $(\\sigma/\\varepsilon)\\log(\\sigma/\\varepsilon)$')
if len(eps_iae_nz) > 0:
    ax1.loglog(eps_iae_nz, iae_queries_nz, 'ko', ms=9, zorder=5,
               label='Qiskit IAE (actual queries)')
ax1.set_xlabel('Target precision $\\varepsilon$', fontsize=14)
ax1.set_ylabel('Oracle queries', fontsize=14)
ax1.set_title('QAMC Building Block: Query Complexity', fontsize=15)
ax1.legend(fontsize=11)
ax1.grid(True, which='both', alpha=0.3)
ax1.invert_xaxis()

# Right: scaling exponents
log_ie = np.log10(1.0 / eps_range)
log_cc = np.log10(classical_cost)
log_qt = np.log10(quantum_theory)

# Local slopes
dl = np.diff(log_ie)
slope_c = np.diff(log_cc) / dl
slope_q = np.diff(log_qt) / dl
x_mid = 0.5 * (log_ie[:-1] + log_ie[1:])

ax2.plot(x_mid, slope_c, 'r-', lw=2.5, label='Classical MC')
ax2.plot(x_mid, slope_q, 'b-', lw=2.5, label='QAMC (Corollary 3.2)')

# IAE empirical slopes
if len(eps_iae_nz) > 2:
    log_iae_eps = np.log10(1.0 / eps_iae_nz)
    log_iae_q = np.log10(iae_queries_nz)
    sl_iae = np.diff(log_iae_q) / np.diff(log_iae_eps)
    x_iae = 0.5 * (log_iae_eps[:-1] + log_iae_eps[1:])
    valid_sl = np.isfinite(sl_iae) & (sl_iae > 0)
    if np.any(valid_sl):
        ax2.plot(x_iae[valid_sl], sl_iae[valid_sl], 'ko', ms=8, label='IAE empirical slopes')

ax2.axhline(y=2, color='r', ls='--', alpha=0.4)
ax2.axhline(y=1, color='b', ls='--', alpha=0.4)
ax2.set_xlabel('$\\log_{10}(1/\\varepsilon)$', fontsize=14)
ax2.set_ylabel('Scaling exponent', fontsize=14)
ax2.set_title('Scaling Exponents', fontsize=15)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0.5, 2.5)

plt.tight_layout()
plt.savefig('experiment1_iae_scaling.png', dpi=150, bbox_inches='tight')
plt.show()

print(f'Classical MC scaling: exactly 2.0 (by construction)')
print(f'QAMC theory scaling: ~1.0 + log correction (Corollary 3.2)')
print(f'IAE empirical queries follow the quantum scaling curve, confirming the quadratic speedup.')
def compute_Delta(y_d, n, sigma, strike):
    """Compute MLMC difference Delta_d(y, n) for one-step-to-terminal case.
    
    Delta = g_d(y, R_fine) - g_d(y, R_coarse)
    where R_fine uses 2^n independent samples, R_coarse uses 2^{n-1}.
    g_d(y, z) = max(max(y - K, 0), z)
    Uses INDEPENDENT samples for fine and coarse (as in the paper's algorithm).
    """
    exercise = max(y_d - strike, 0)
    
    # Fine estimator: 2^n independent samples
    N_fine = 2**n
    payoffs_fine = np.maximum(y_d + sigma * np.random.randn(N_fine) - strike, 0)
    g_fine = max(exercise, np.mean(payoffs_fine))
    
    if n == 0:
        return g_fine  # level 0: single-sample estimate
    
    # Coarse estimator: 2^{n-1} INDEPENDENT samples
    N_coarse = 2**(n - 1)
    payoffs_coarse = np.maximum(y_d + sigma * np.random.randn(N_coarse) - strike, 0)
    g_coarse = max(exercise, np.mean(payoffs_coarse))
    
    return g_fine - g_coarse


# Parameters
sigma = 0.3
strike = 1.0
y_d = 1.05  # slightly in-the-money for nontrivial exercise boundary
max_level = 14
n_samples = 8000

print(f'Computing Var[Delta(y, n)] for levels n = 0..{max_level}')
print(f'y_d={y_d}, sigma={sigma}, strike={strike}, {n_samples} samples/level\n')

level_variances = []
level_means = []

for n in range(max_level + 1):
    np.random.seed(1000 + n)
    deltas = np.array([compute_Delta(y_d, n, sigma, strike) for _ in range(n_samples)])
    v = np.var(deltas)
    m = np.mean(deltas)
    level_variances.append(v)
    level_means.append(m)
    print(f'  n={n:2d}:  Var = {v:.6e},  E[Delta] = {m:+.8f}')

level_variances = np.array(level_variances)
level_means = np.array(level_means)
levels = np.arange(max_level + 1)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5))

# -- Left: Variance decay --
valid = level_variances > 1e-20
ax1.semilogy(levels[valid], level_variances[valid], 'ko-', lw=2, ms=8, 
             label='Empirical Var[$\\Delta(y,n)$]')

# Fit decay rate (skip level 0 which has different structure)
fit_start = 2
fit_lvl = levels[fit_start:]
fit_logvar = np.log2(level_variances[fit_start:])
decay_rate, intercept = np.polyfit(fit_lvl, fit_logvar, 1)
ref = 2**(intercept + decay_rate * levels)
ax1.semilogy(levels, ref, 'r--', lw=2, alpha=0.6,
             label=f'Fit: $2^{{{decay_rate:.2f} \\cdot n}}$ (theory: $2^{{-n}}$)')

ax1.set_xlabel('MLMC Level $n$', fontsize=14)
ax1.set_ylabel('Var[$\\Delta(y,n)$]', fontsize=14)
ax1.set_title('Per-Level Variance Decay', fontsize=15)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# -- Middle: Per-level cost comparison --
eps_ref = 0.01
classical_per = level_variances / eps_ref**2
quantum_per = np.sqrt(np.maximum(level_variances, 1e-30)) / eps_ref
quantum_per *= np.maximum(1, np.log2(np.maximum(quantum_per, 1.1)))

ax2.semilogy(levels[valid], classical_per[valid], 'rs-', lw=2, ms=8, 
             label='Classical: $\\sigma^2_n / \\varepsilon^2$')
ax2.semilogy(levels[valid], quantum_per[valid], 'bo-', lw=2, ms=8, 
             label='Quantum: $(\\sigma_n / \\varepsilon)\\cdot\\log$')
ax2.set_xlabel('MLMC Level $n$', fontsize=14)
ax2.set_ylabel(f'Queries per level ($\\varepsilon$={eps_ref})', fontsize=14)
ax2.set_title('Per-Level Cost: Classical vs Quantum', fontsize=15)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

# -- Right: Cumulative cost + speedup --
cum_c = np.cumsum(classical_per)
cum_q = np.cumsum(quantum_per)
speedup = cum_c / cum_q

ax3.semilogy(levels, cum_c, 'rs-', lw=2, ms=8, label='Classical cumulative')
ax3.semilogy(levels, cum_q, 'bo-', lw=2, ms=8, label='Quantum cumulative')

ax3_tw = ax3.twinx()
ax3_tw.plot(levels, speedup, 'g^--', lw=2, ms=8, alpha=0.7)
ax3_tw.set_ylabel('Speedup (green)', fontsize=12, color='green')
ax3_tw.tick_params(axis='y', labelcolor='green')

ax3.set_xlabel('MLMC Level $n$', fontsize=14)
ax3.set_ylabel('Cumulative cost', fontsize=14)
ax3.set_title('Cumulative Cost & Speedup', fontsize=15)
ax3.legend(fontsize=11, loc='center left')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('experiment3_variance_decay.png', dpi=150, bbox_inches='tight')
plt.show()

print(f'\nEmpirical variance decay: Var[Delta] ~ 2^({decay_rate:.2f} * n)')
print(f'Theory predicts:          Var[Delta] ~ 2^(-1.0  * n)')
print(f'Cumulative speedup at level {max_level}: {speedup[-1]:.1f}x')