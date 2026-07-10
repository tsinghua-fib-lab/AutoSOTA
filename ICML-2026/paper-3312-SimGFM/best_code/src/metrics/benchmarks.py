import time
from typing import Dict, Optional, List


# Guacamol (optional)
try:  # pragma: no cover - optional dependency
    from guacamol.assess_distribution_learning import kl_divergence as guacamol_kl_div  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    try:
        from guacamol.distribution_matching import kl_divergence as guacamol_kl_div  # type: ignore
    except Exception:
        guacamol_kl_div = None  # type: ignore


def compute_guacamol_kl(generated_smiles, ref_train):
    """Compute KL divergence according to Guacamol helper if available.

    Returns a dict {"kl_div": value} or -1.0 on failure. Prints runtime.
    """
    if guacamol_kl_div is None:
        print("Guacamol not installed or KL API unavailable; skipping KL divergence.")
        return -1.0
    start = time.time()
    try:
        kl_val = guacamol_kl_div(generated_smiles, ref_train)
    except Exception as e:
        print(f"Error computing Guacamol KL divergence: {e}")
        return -1.0
    finally:
        end = time.time()
        print("Guacamol KL computation time:", end - start)

    if isinstance(kl_val, dict):
        if "kl_div" in kl_val:
            return {"kl_div": float(kl_val["kl_div"]) }
        out: Dict[str, float] = {}
        for k, v in kl_val.items():
            try:
                out[str(k)] = float(v)
            except Exception:
                continue
        return out if out else -1.0
    try:
        return {"kl_div": float(kl_val)}
    except Exception:
        return -1.0


