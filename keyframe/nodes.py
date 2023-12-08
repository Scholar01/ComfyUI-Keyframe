import numpy as np

from .interface import KeyframePartGroup, KeyframePart, ModelInjectParam
from .sampling import keyframe_sample_factory

from .util import inject_model
from .samples import inject_samples
import comfy.sample as comfy_sample

inject_samples()
comfy_sample.sample = keyframe_sample_factory(comfy_sample.sample)


class KeyframePartNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "batch_index": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1}),
                "denoise": ("FLOAT", {"default": 0.1, "min": 0.001, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "part": ("LATENT_KEYFRAME_PART",),
            }
        }

    RETURN_TYPES = ("LATENT_KEYFRAME_PART",)
    RETURN_NAMES = ("part",)
    FUNCTION = "load_keyframe_part"

    CATEGORY = "KeyframePart"

    def load_keyframe_part(self, image, batch_index, denoise, part=None):
        if not part:
            part = KeyframePartGroup()
        part = part.clone()
        keyframe = KeyframePart(batch_index, image, denoise)
        part.add(keyframe)
        return (part,)


class KeyframeInterpolationPartNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "batch_index_from": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1}),
                "batch_index_to": ("INT", {"default": 4, "min": 1, "max": 9999, "step": 1}),
                "denoise_from": ("FLOAT", {"default": 0.1, "min": 0.001, "max": 1.0, "step": 0.01}),
                "denoise_to": ("FLOAT", {"default": 1.0, "min": 0.001, "max": 1.0, "step": 0.01}),
                "interpolation": (["linear", "ease-in", "ease-out", "ease-in-out"],),
            },
            "optional": {
                "part": ("LATENT_KEYFRAME_PART",),
            }
        }

    RETURN_TYPES = ("LATENT_KEYFRAME_PART",)
    RETURN_NAMES = ("part",)
    FUNCTION = "load_keyframe_part"

    CATEGORY = "KeyframeInterpolationPartNode"

    def load_keyframe_part(self,
                           image,
                           batch_index_from,
                           batch_index_to,
                           denoise_from,
                           denoise_to,
                           interpolation,
                           part=None):
        if batch_index_from >= batch_index_to:
            raise ValueError("batch_index_from must be less than batch_index_to")

        if not part:
            part = KeyframePartGroup()
        part = part.clone()
        current_group = KeyframePartGroup()

        steps = batch_index_to - batch_index_from
        diff = denoise_to - denoise_from
        if interpolation == "linear":
            weights = np.linspace(denoise_from, denoise_to, steps)
        elif interpolation == "ease-in":
            index = np.linspace(0, 1, steps)
            weights = diff * np.power(index, 2) + denoise_from
        elif interpolation == "ease-out":
            index = np.linspace(0, 1, steps)
            weights = diff * (1 - np.power(1 - index, 2)) + denoise_from
        elif interpolation == "ease-in-out":
            index = np.linspace(0, 1, steps)
            weights = diff * ((1 - np.cos(index * np.pi)) / 2) + denoise_from

        for i in range(steps):
            keyframe = KeyframePart(batch_index_from + i, image, float(weights[i]))
            current_group.add(keyframe)

        # replace values with prev_latent_keyframes
        for latent_keyframe in part.keyframes:
            current_group.add(latent_keyframe)

        return (current_group,)


class KeyframeApplyNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "latent": ("LATENT",),
                "part_group": ("LATENT_KEYFRAME_PART",),
                "vae": ("VAE",),
            }
        }

    RETURN_TYPES = ("MODEL", "LATENT",)
    RETURN_NAMES = ("model", "latent",)
    FUNCTION = "apply_latent_keyframe"

    CATEGORY = "LatentKeyframeApply"

    @staticmethod
    def vae_encode_crop_pixels(pixels):
        x = (pixels.shape[1] // 8) * 8
        y = (pixels.shape[2] // 8) * 8
        if pixels.shape[1] != x or pixels.shape[2] != y:
            x_offset = (pixels.shape[1] % 8) // 2
            y_offset = (pixels.shape[2] % 8) // 2
            pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
        return pixels

    def encode(self, vae, pixels):
        pixels = self.vae_encode_crop_pixels(pixels)
        t = vae.encode(pixels[:, :, :, :3])
        return t

    def apply_latent_keyframe(self, model, latent, part_group: KeyframePartGroup, vae):
        # 预处理latent,把图片替换进去
        for part in part_group.keyframes:
            latent['samples'][part.batch_index] = self.encode(vae, part.image)
            print(f"apply keyframe {part.batch_index}:{part.denoise}")

        # 注入参数,后续处理

        inject_param = ModelInjectParam(part_group, latent)
        inject_model(model.model, inject_param)

        return (model, latent,)


# NODE MAPPING
NODE_CLASS_MAPPINGS = {
    # Keyframes
    "KeyframePart": KeyframePartNode,
    "KeyframeInterpolationPart": KeyframeInterpolationPartNode,
    "KeyframeApply": KeyframeApplyNode,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Keyframes
    "KeyframePart": "Keyframe Part",
    "KeyframeInterpolationPart": "Keyframe Interpolation Part",
    "KeyframeApply": "Keyframe Apply",
}
