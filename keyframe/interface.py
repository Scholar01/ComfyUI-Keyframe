from dataclasses import dataclass, field
from typing import Union

import torch


class KeyframePart:
    def __init__(self, batch_index: int, image: torch.Tensor, denoise: float) -> None:
        self.batch_index = batch_index
        self.denoise = denoise
        self.image = image


class KeyframePartGroup:
    def __init__(self) -> None:
        self.keyframes: list[KeyframePart] = []

    def add(self, keyframe: KeyframePart) -> None:
        added = False
        for i in range(len(self.keyframes)):
            if self.keyframes[i].batch_index == keyframe.batch_index:
                self.keyframes[i] = keyframe
                added = True
                break
        if not added:
            self.keyframes.append(keyframe)
        self.keyframes.sort(key=lambda k: k.batch_index)

    def get_index(self, index: int) -> Union[KeyframePart, None]:
        try:
            return self.keyframes[index]
        except IndexError:
            return None

    def __getitem__(self, index) -> KeyframePart:
        return self.keyframes[index]

    def is_empty(self) -> bool:
        return len(self.keyframes) == 0


@dataclass
class ModelInjectParam:
    keyframe_part_group: KeyframePartGroup
    latent: dict = field(default_factory=dict, repr=False)
    seed: int = 0
    steps: int = 0
    scheduler: str = 'normal'
    denoise: float = 0
    noise: torch.Tensor = field(default=None, repr=False)

    def reset(self):
        self.seed: int = 0
        self.steps: int = 0
        self.scheduler: str = 'normal'
        self.denoise: float = 0
        self.noise: torch.Tensor = None
