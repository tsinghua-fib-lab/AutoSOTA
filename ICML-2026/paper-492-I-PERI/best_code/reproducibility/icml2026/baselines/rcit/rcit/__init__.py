# SPDX-FileCopyrightText: 2024 Pôle d'Expertise de la Régulation Numérique <contact.peren@finances.gouv.fr>
#
# SPDX-License-Identifier: MIT

import importlib.metadata
from typing import Literal

# __version__ = importlib.metadata.version(__package__)
ApproxMethod = Literal["lpd4", "chi2", "gamma", "hbe"]
