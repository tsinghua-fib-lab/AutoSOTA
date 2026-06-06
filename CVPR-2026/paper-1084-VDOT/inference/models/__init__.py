# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from . import utils

try:
    from . import ltx
except ImportError as e:
    print("Warning: failed to importing 'ltx'. Please install its dependencies with:")
    print("pip install ltx-video@git+https://github.com/Lightricks/LTX-Video@ltx-video-0.9.1 sentencepiece --no-deps")

try:
    from . import wan
except ImportError as e:
    print("Warning: failed to importing 'wan'. Please install its dependencies with:")
    print("pip install wan@git+https://github.com/Wan-Video/Wan2.1")
