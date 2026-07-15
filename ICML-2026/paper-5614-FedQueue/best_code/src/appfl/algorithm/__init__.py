"""
Warning: Importing algorithms directly from this module (e.g., `from appfl.algorithm import ServerFedAvg`)
is deprecated and will be removed in future versions.
Please use:
    - `appfl.algorithm.aggregator` for server aggregation algorithms,
    - `appfl.algorithm.trainer` for client local training algorithms,
    - `appfl.algorithm.scheduler` for server scheduling algorithms

This module provides a base class to define the functions required to implement PPFL algorithms.
"""

from .legacy.fl_base import BaseClient, BaseServer
from .legacy.ppfl_base import PPFLClient, PPFLServer
from .legacy.client_optimizer import ClientOptim
from .legacy.client_step_optimizer import ClientStepOptim
from .legacy.globus_compute_client_optimizer import GlobusComputeClientOptim
from .legacy.globus_compute_client_step_optimizer import GlobusComputeClientStepOptim
from .legacy.personalized_client_optimizer import PersonalizedClientOptim
from .legacy.personalized_client_step_optimizer import PersonalizedClientStepOptim
from .legacy.server_fed_avg import ServerFedAvg
from .legacy.server_fed_avgmom import ServerFedAvgMomentum
from .legacy.server_fed_adagrad import ServerFedAdagrad
from .legacy.server_fed_adam import ServerFedAdam
from .legacy.server_fed_yogi import ServerFedYogi
from .legacy.iceadmm import ICEADMMClient, ICEADMMServer
from .legacy.iiadmm import IIADMMClient, IIADMMServer
from .legacy.server_fed_asynchronous import ServerFedAsynchronous
from .legacy.server_fed_buffer import ServerFedBuffer
from .legacy.server_fed_compass import ServerFedCompass
from .legacy.server_fed_compass_mom import ServerFedCompassMom
from .legacy.server_fed_compass_nova import ServerFedCompassNova
from .legacy.scheduler_compass import (
    SchedulerCompass,
    SchedulerCompassGlobusCompute,
    SchedulerCompassMPI,
)
from .legacy.scheduler_dummy import SchedulerDummy

__all__ = [
    "BaseClient",
    "BaseServer",
    "PPFLClient",
    "PPFLServer",
    "ClientOptim",
    "ClientStepOptim",
    "GlobusComputeClientOptim",
    "GlobusComputeClientStepOptim",
    "PersonalizedClientOptim",
    "PersonalizedClientStepOptim",
    "ServerFedAvg",
    "ServerFedAvgMomentum",
    "ServerFedAdagrad",
    "ServerFedAdam",
    "ServerFedYogi",
    "ICEADMMClient",
    "ICEADMMServer",
    "IIADMMClient",
    "IIADMMServer",
    "ServerFedAsynchronous",
    "ServerFedBuffer",
    "ServerFedCompass",
    "ServerFedCompassMom",
    "ServerFedCompassNova",
    "SchedulerCompass",
    "SchedulerCompassGlobusCompute",
    "SchedulerCompassMPI",
    "SchedulerDummy",
]
