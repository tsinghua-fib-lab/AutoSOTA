# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Some other utility functions."""


import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import absl.logging
import tensorflow as tf


def setup_tf():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False


class EasyDict(dict):
    """Custom dictionary that allows to access dict values as attributes."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def get_fn(general_params, privacy_params, paramss, find_next=False):
    """Generate the output file name given parameters."""
    dire = '_'.join([k + str(v) for k, v in general_params.items()]) + '/'

    fn = []
    if not privacy_params.dpsgd:
        fn += ['nonpriv']
    else:
        fn += ['_'.join(
            [k + str(privacy_params[k]) for k in sorted(privacy_params) if k not in ['dpsgd', 'completion'] and privacy_params[k]])]
        if privacy_params.completion:
            fn += ['completion']

    for params in paramss:
        s = []
        for k in sorted(params):
            if not params[k]:
                continue
            if isinstance(params[k], bool):
                s.append(k)
            else:
                s.append(k + str(params[k]))
        s = '_'.join(s)
        if s is not None and len(s) > 0:
            fn.append(s)

    fn = dire + '_'.join(fn)
    if not find_next:
        return fn

    if not os.path.exists(fn):
        return fn
    i = 1
    while os.path.exists(fn + '_{}'.format(i)):
        i += 1
    fn = fn + '_{}'.format(i)
    return fn
