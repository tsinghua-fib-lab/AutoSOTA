from typing import Dict, List
import torch
import logging

pylogger = logging.getLogger(__name__)

def cycle_error(
    translator,                    
    spaces: Dict[str, torch.Tensor],
    splits: List,
    *,        
    atol: float = 1e-2,
    rtol: float = 1e-2,
):
    """
    Check that mapping embeddings around `cycle` and back to the start
    returns (approximately) the original batch.

    Parameters
    ----------
    translator : fitted translator with .transform()
    spaces     : dict  {"train": {enc: (N,d)}, "test": {...}, ...}
    cycle      : list  encoder names, e.g. ["clip", "bert", "vit"]
                 The function will do clip → bert → vit → clip.
    atol, rtol : tolerances passed to torch.allclose

    Returns
    -------
    err   : (N,)  L2-norm per sample between original and reconstructed
    ok    : bool  torch.allclose(...) over the whole batch
    """
    assert translator._fitted, "Translator must be fitted first."

    for split in splits:
        cycle=list(spaces[split].keys())
        assert len(cycle) >= 2, "Cycle needs at least two encoders."
        start = cycle[0]
        x0 = spaces[split][start]

        x = x0
        z0 = translator._zscore(translator._pad(name=start, x=x0), name=start)
        z = translator._zscore(translator._pad(name=start, x=x0), name=start)

        for src, tgt in zip(cycle, cycle[1:] + [start]):
            z = z @ translator.R_out[src] @ translator.R_out[tgt].T
            
        err = (z0 - z).norm(dim=1) # compare in z-scored spaces
        ok  = torch.allclose(z0, z, atol=atol, rtol=rtol)
        assert ok, f"Cycle error for {split}: {err} far from 0? based on tolerance {atol}"
        logging.info(f"Your translator has almost 0 cycle-error based on tolerance {atol}!")

def cycle_consistency(translator, 
                    spaces: Dict[str, torch.Tensor],
                    splits: List,
                ):

    for split in splits:
        encoders = list(spaces[split].keys())
    
        pylogger.info(f"Computing cyclic composition for {split}")
        cyclic_composition = None
        for i in range(len(encoders)):
            src = encoders[i]
            tgt = encoders[(i + 1) % len(encoders)]  # wrap around cyclically
            pylogger.info(f"Pairwise map: {src} -> {tgt}")
            mapping = translator.pairwise_map(src=src, tgt=tgt)

            cyclic_composition = mapping if cyclic_composition is None else cyclic_composition @ mapping

    assert torch.allclose(cyclic_composition, torch.eye(cyclic_composition.shape[0]).to(cyclic_composition.device), atol=1e-1), "Cyclic composition does not yield identity matrix"
    logging.info("Your translator is cycle-consistent!")