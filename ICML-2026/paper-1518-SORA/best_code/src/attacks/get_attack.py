from .fgsm import fgsm
from .fgm import fgm
from .trades import trades
from .fgsm_rs import fgsm_rs
from .fgm_rs import fgm_rs
from .grad_align import grad_align
from .atas import atas
from .nuclear import nuclear
from .zerograd import zero_grad
from .multigrad import multi_grad
from .nfgsm import nfgsm
from .aaer import fgsm as fgsm_aae
from .elle import elle
from .sora import sora
from .pgd import pgd

def get_attack(attack_name: str):
    """
    Get the attack for the given attack name.
    
    Args:
        attack_name (str): Name of the attack to get.
    """
    
    match attack_name:
        case "Benign":
            return None
        case "FGSM":
            return fgsm
        case "FGM":
            return fgm
        case "TRADES":
            return trades
        case "FGSM-RS":
            return fgsm_rs
        case "FGM-RS":
            return fgm_rs
        case "GradAlign":
            return grad_align
        case "ATAS":
            return atas
        case "NuAT":
            return nuclear
        case "ZeroGrad":
            return zero_grad
        case "MultiGrad":
            return multi_grad
        case "NFGSM":
            return nfgsm
        case "AAER":
            return fgsm_aae
        case "ELLE":
            return elle
        case "SORA":
            return sora
        case "PGD":
            return pgd
        case "PGD2":
            return pgd
        case _:
            raise ValueError('Invalid Attack!')
        
