# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from .depth import DepthAnnotator, DepthVideoAnnotator, DepthV2VideoAnnotator
from .flow import FlowAnnotator, FlowVisAnnotator
from .frameref import FrameRefExtractAnnotator, FrameRefExpandAnnotator
from .gdino import GDINOAnnotator, GDINORAMAnnotator
from .gray import GrayAnnotator, GrayVideoAnnotator
from .inpainting import InpaintingAnnotator, InpaintingVideoAnnotator
from .layout import LayoutBboxAnnotator, LayoutMaskAnnotator, LayoutTrackAnnotator
from .maskaug import MaskAugAnnotator
from .outpainting import OutpaintingAnnotator, OutpaintingInnerAnnotator, OutpaintingVideoAnnotator, OutpaintingInnerVideoAnnotator
from .pose import PoseBodyFaceAnnotator, PoseBodyFaceVideoAnnotator, PoseAnnotator, PoseBodyVideoAnnotator, PoseBodyAnnotator
from .pose2 import PoseBodyFaceAnnotator2, PoseBodyFaceVideoAnnotator2, PoseAnnotator2, PoseBodyVideoAnnotator2, PoseBodyAnnotator2
from .ram import RAMAnnotator
from .salient import SalientAnnotator, SalientVideoAnnotator
from .sam import SAMImageAnnotator
from .sam2 import SAM2ImageAnnotator, SAM2VideoAnnotator, SAM2SalientVideoAnnotator, SAM2GDINOVideoAnnotator
from .scribble import ScribbleAnnotator, ScribbleVideoAnnotator
from .face import FaceAnnotator
from .subject import SubjectAnnotator
from .common import PlainImageAnnotator, PlainMaskAnnotator, PlainMaskAugAnnotator, PlainMaskVideoAnnotator, PlainVideoAnnotator, PlainMaskAugVideoAnnotator, PlainMaskAugInvertAnnotator, PlainMaskAugInvertVideoAnnotator, ExpandMaskVideoAnnotator
from .prompt_extend import PromptExtendAnnotator
from .composition import CompositionAnnotator, ReferenceAnythingAnnotator, AnimateAnythingAnnotator, SwapAnythingAnnotator, ExpandAnythingAnnotator, MoveAnythingAnnotator
from .mask import MaskDrawAnnotator
from .canvas import RegionCanvasAnnotator