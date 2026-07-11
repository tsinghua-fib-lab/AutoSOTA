# Copyright 2023 Multiscale Modeling of Fluid Materials, TU Munich
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def datafiles(request):
    """Returns the corresponding datafiles folder."""
    # Path where pytest is executed can be overridden
    base_path = os.environ.get("PYTEST_PATH", ".")
    base_path = Path(base_path).absolute() / "tests"

    rel_path = Path(request.fspath).relative_to(base_path)
    data_path = base_path / "data" / rel_path.parent / rel_path.stem

    return data_path
