import math

import torch

from comfy.samplers import calculate_sigmas_scheduler

KEYFRAME_INJECTED_ATTR = "keyframe_injected"


def inject_model(model, inject_param):
    # 注入模型参数
    setattr(model, KEYFRAME_INJECTED_ATTR, inject_param)
    return model


def is_injected_model(model):
    return hasattr(model, KEYFRAME_INJECTED_ATTR)


def get_injected_model(model):
    return getattr(model, KEYFRAME_INJECTED_ATTR)


def clear_injected_model(model):
    if is_injected_model(model):
        delattr(model, KEYFRAME_INJECTED_ATTR)




def max_denoise(model_wrap, sigmas):
    max_sigma = float(model_wrap.inner_model.model_sampling.sigma_max)
    sigma = float(sigmas[0])
    return math.isclose(max_sigma, sigma, rel_tol=1e-05) or sigma > max_sigma


def generate_sigmas(real_model, x, origin_sigmas, scheduler, steps, part_group, device):
    batch_size = x.shape[0]
    new_sigmas = origin_sigmas.unsqueeze(0).repeat(batch_size, 1)

    for part in part_group:
        if part.denoise is None or part.denoise > 0.9999:
            new_sigmas[part.batch_index] = calculate_sigmas_scheduler(real_model, scheduler, steps).to(device)
        else:
            new_steps = int(steps / part.denoise)
            sigmas = calculate_sigmas_scheduler(real_model, scheduler, new_steps).to(device)
            new_sigmas[part.batch_index] = sigmas[-(steps + 1):]
    return new_sigmas


def generate_noise(model_wrap, sigmas, noise):
    if max_denoise(model_wrap, sigmas):
        n = noise * torch.sqrt(1.0 + sigmas[0] ** 2.0)
    else:
        n = noise * sigmas[0]
    return n


def get_ancestral_step(sigma_from: torch.Tensor, sigma_to: torch.Tensor, eta: float = 1.) -> (
        torch.Tensor, torch.Tensor):
    if not eta:
        return sigma_to, torch.zeros_like(sigma_to)
    sigma_up = torch.min(sigma_to,
                         eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5)
    sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5

    return sigma_down, sigma_up
