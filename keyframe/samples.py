import torch
from tqdm.auto import trange

import comfy.samplers
from comfy.k_diffusion import sampling as k_diffusion_sampling
from comfy.k_diffusion.sampling import to_d, default_noise_sampler
from .util import  is_injected_model, get_injected_model, generate_sigmas, generate_noise, get_ancestral_step

CUSTOM_SAMPLERS = [
    'k_euler', 'k_euler_a', 'k_lcm'
]


def inject_samples():
    comfy.samplers.SAMPLER_NAMES.extend(CUSTOM_SAMPLERS)
    k_diffusion_sampling.sample_k_euler = sample_k_euler
    k_diffusion_sampling.sample_k_euler_a = sample_k_euler_a
    k_diffusion_sampling.sample_k_lcm = sample_k_lcm
    print(f'Injected samplers: {CUSTOM_SAMPLERS}')


def get_sigmas_noise(model_wrap, x, noise, latent_image, sigmas, scheduler, steps, part_group):
    sigmas = generate_sigmas(model_wrap.inner_model, x, sigmas, scheduler, steps, part_group, sigmas.device)
    noise = noise.to(x.device)
    latent_image = latent_image.to(x.device)
    for i in range(noise.shape[0]):
        noise[i] = generate_noise(model_wrap, sigmas[i], noise[i])

    if latent_image is not None:
        latent_image = model_wrap.inner_model.process_latent_in(latent_image)
        noise += latent_image
    return sigmas, noise


@torch.no_grad()
def sample_k_euler(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0.,
                   s_tmax=float('inf'), s_noise=1.):
    model_wrap = model.inner_model
    real_model = model_wrap.inner_model

    if not is_injected_model(real_model):
        raise Exception("model is not injected,please use LatentKeyframeApply node to inject model")

    inject_param = get_injected_model(real_model)
    latent_image = inject_param.latent['samples']

    sigmas, noise = get_sigmas_noise(model_wrap, x, inject_param.noise, latent_image, sigmas, inject_param.scheduler,
                                     inject_param.steps, inject_param.keyframe_part_group)

    extra_args = {} if extra_args is None else extra_args
    s_tmin = s_tmin * x.new_ones([sigmas.shape[0]])
    s_tmax = s_tmax * x.new_ones([sigmas.shape[0]])
    gammas = s_tmax * x.new_ones([sigmas.shape[0]])
    for i in trange(sigmas.shape[1] - 1, disable=disable):
        for j, sigma in enumerate(sigmas):
            gammas[j] = min(s_churn / (sigmas.shape[0] - 1), 2 ** 0.5 - 1) if s_tmin[j] <= sigma[i] <= s_tmax[j] else 0.
        sigma_hat = sigmas[:, i] * (gammas + 1)

        for j, gamma in enumerate(gammas):
            if gamma > 0:
                eps = torch.randn_like(noise) * s_noise
                noise[j] = noise[j] + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5

        denoised = model(noise, sigma_hat, **extra_args)
        d = to_d(noise, sigma_hat, denoised)
        if callback is not None:
            callback({'x': noise, 'i': i, 'sigma': sigma_hat, 'denoised': denoised})
        dt = sigmas[:, i + 1] - sigma_hat
        noise += d * dt.view(d.shape[0], 1, 1, 1)
    return noise


@torch.no_grad()
def sample_k_euler_a(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1.,
                     noise_sampler=None):

    model_wrap = model.inner_model
    real_model = model_wrap.inner_model

    if not is_injected_model(real_model):
        raise Exception("model is not injected,please use LatentKeyframeApply node to inject model")

    inject_param = get_injected_model(real_model)
    latent_image = inject_param.latent['samples']

    sigmas, noise = get_sigmas_noise(model_wrap, x, inject_param.noise, latent_image, sigmas, inject_param.scheduler,
                                     inject_param.steps, inject_param.keyframe_part_group)

    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(noise) if noise_sampler is None else noise_sampler

    for i in trange(sigmas.shape[1] - 1, disable=disable):
        s_in = sigmas[:, i]
        denoised = model(noise, s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[:, i], sigmas[:, i + 1], eta=eta)
        if callback is not None:
            callback({'x': noise, 'i': i, 'sigma': s_in, 'denoised': denoised})
        d = to_d(noise, s_in, denoised)
        # Euler method
        dt = sigma_down - s_in
        noise += d * dt.view(d.shape[0], 1, 1, 1)
        for j, sigma in enumerate(sigmas):
            if sigma[i + 1] > 0:
                noise[j] = noise[j] + noise_sampler(sigma[i], sigma[i + 1])[j] * s_noise * sigma_up[j]
    return noise


@torch.no_grad()
def sample_k_lcm(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None):
    
    model_wrap = model.inner_model
    real_model = model_wrap.inner_model

    if not is_injected_model(real_model):
        raise Exception("model is not injected,please use LatentKeyframeApply node to inject model")
    
    inject_param = get_injected_model(real_model)
    latent_image = inject_param.latent['samples']

    sigmas, noise = get_sigmas_noise(model_wrap, x, inject_param.noise, latent_image, sigmas, inject_param.scheduler,
                                     inject_param.steps, inject_param.keyframe_part_group)

    
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(noise) if noise_sampler is None else noise_sampler

    for i in trange(sigmas.shape[1] - 1, disable=disable):
        # s_in = x.new_ones([x.shape[0]])
        s_in = sigmas[:, i]
        denoised = model(noise, s_in, **extra_args)
        if callback is not None:
            callback({'x': noise, 'i': i, 'sigma': s_in, 'sigma_hat':s_in, 'denoised': denoised})
        
   
        noise = denoised

        
        for j, sigma in enumerate(sigmas):

            if sigma[i + 1] > 0:
                noise[j] = noise[j] + sigma[i + 1] * noise_sampler(sigma[i], sigma[i + 1])[j]
    return noise

