import torch
from typing import Dict, Optional
from ultralytics.nn.tasks import SegmentationModel, DetectionModel, ClassificationModel
from ultralytics.yolo.utils import LOGGER
__all__ = ["is_dynamic_axes"]

def is_dynamic_axes(model: torch.nn.Module, dynamic: bool) -> Optional[Dict[int, str]]:
    if dynamic:
        LOGGER.info("Exporting model to ONNX with dynamic axes... ")
        dynamic = {'images': {0: 'batch', 2: 'height', 3: 'width'}}  # shape(1, 3, 640, 640)
        if isinstance(model, SegmentationModel):
            raise NotImplementedError()
        elif isinstance(model, DetectionModel):
            dynamic['output0'] = {0: 'batch', 1: 'anchors'}  # shape (1, 84, 8400)
            return dynamic
        elif isinstance(model, ClassificationModel):
            raise NotImplementedError()
        else:
            raise ValueError("ERROR: Unknown model type. "
                             "Only SegmentationModel, DetectionModel, "
                             "and ClassificationModel are supported.")
    return None

