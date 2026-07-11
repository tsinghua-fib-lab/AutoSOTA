"""Generate simple rBergomi dataset with clean signature window alignment."""
import numpy as np
from pathlib import Path

N_PATHS = 15000
T_STEPS = 129   # 129 gives effective=129 with window=16: (129-1)%16=0
T_YEARS = 1.0
H = 0.07
XI0 = 0.235 ** 2
ETA = 1.5
SEED = 42

def generate_simple_rbergomi(n_paths=15000, n_steps=129, seed=42):
    rng = np.random.default_rng(seed)
    dt = T_YEARS / n_steps

    inc = np.sqrt(dt) * rng.standard_normal((n_paths, n_steps - 1), dtype=np.float64)
    w_driver = np.zeros((n_paths, n_steps), dtype=np.float64)
    w_driver[:, 1:] = np.cumsum(inc, axis=1)
    driver = w_driver[:, :, np.newaxis].astype(np.float32)

    t = np.linspace(0, T_YEARS, n_steps, dtype=np.float64)
    m = 2 * n_steps
    c = np.zeros(m, dtype=np.float64)
    for k in range(n_steps):
        c[k] = 0.5 * (t[k] ** (2 * H) + t[0] ** (2 * H) - abs(t[k] - t[0]) ** (2 * H))
    for k in range(n_steps, m):
        c[k] = c[m - k]

    eigvals = np.fft.fft(c).real
    eigvals = np.maximum(eigvals, 1e-12)

    noise_real = rng.standard_normal((n_paths, m), dtype=np.float64)
    noise_imag = rng.standard_normal((n_paths, m), dtype=np.float64)

    sqrt_eig = np.sqrt(eigvals / m)
    z = noise_real + 1j * noise_imag
    z_scaled = z * sqrt_eig[None, :]
    fbm = np.fft.ifft(z_scaled, axis=1).real[:, :n_steps]

    a = H - 0.5
    Y = np.sqrt(2 * a + 1) * fbm * (t[None, :] ** (a + 0.5))

    log_var = ETA * Y - 0.5 * ETA**2 * t[None, :] ** (2 * a + 1)
    variance = XI0 * np.exp(log_var)
    variance = np.maximum(variance, 1e-12)

    log_price = np.zeros((n_paths, n_steps), dtype=np.float64)
    sigma = np.sqrt(variance[:, :-1])
    dW_log = np.diff(w_driver, axis=1)
    increments = sigma * dW_log - 0.5 * sigma**2 * dt
    log_price[:, 1:] = np.cumsum(increments, axis=1)

    return driver, log_price.astype(np.float32)


def main():
    output_dir = Path("/repo/data/rough_volatility")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "simple_rbergomi_data.npz"

    print(f"Generating simple rBergomi: {N_PATHS} paths x {T_STEPS} steps")
    driver, log_price = generate_simple_rbergomi(N_PATHS, T_STEPS, SEED)

    print(f"Driver: {driver.shape}, dtype={driver.dtype}")
    print(f"Log-price: {log_price.shape}, dtype={log_price.dtype}")
    print(f"Log-price range: [{log_price.min():.4f}, {log_price.max():.4f}]")

    np.savez_compressed(
        output_path,
        driver=driver,
        log_price=log_price,
        variance=np.zeros(0),
        dt=float(T_YEARS / T_STEPS)
    )

    data = np.load(output_path)
    fs_mb = output_path.stat().st_size / 1024 / 1024
    print(f"Saved {output_path} ({fs_mb:.1f} MB)")
    print(f"Keys: {list(data.keys())}")


if __name__ == "__main__":
    main()
