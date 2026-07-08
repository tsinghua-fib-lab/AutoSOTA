from .geodesics.geodesic_solver_numpy import *
from .geodesics.geodesic_solver_torch import * 
from .geodesics.exponential_numpy import KExponentialNumpy
from .geodesics.exponential_torch import KExponentialTorch
from .geodesics.metrics import *

from .latentrepresentation.training_latent_representation import *
from .latentrepresentation.network import *
from .latentrepresentation.latent_point_cloud import *
from .latentrepresentation.utils import *

def main() -> None:
    print("Hello from latentgeodesics!")

