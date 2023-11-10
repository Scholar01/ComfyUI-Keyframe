from typing import Callable

from comfy.model_patcher import ModelPatcher
from .util import is_injected_model, print, get_injected_model


def keyframe_sample_factory(orig_comfy_sample: Callable) -> Callable:
    def keyframe_sample(model: ModelPatcher, *args, **kwargs):
        if not is_injected_model(model.model):
            return orig_comfy_sample(model, *args, **kwargs)
        inject_param = get_injected_model(model.model)
        try:
            inject_param.reset()

            inject_param.noise = args[0]
            inject_param.steps = args[1]
            inject_param.scheduler = args[4]
            inject_param.denoise = kwargs.get('denoise', None)
            inject_param.seed = kwargs.get('seed', None)
            return orig_comfy_sample(model, *args, **kwargs)
        finally:
            inject_param.reset()

    return keyframe_sample
