from typing import Callable
from abc import abstractmethod

import torch


class ReversableTransform:

    @abstractmethod
    def __call__(self, tensor):
        return NotImplemented

    @abstractmethod
    def inverse(self) -> Callable:
        return NotImplemented


class CustomNormalize(ReversableTransform):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return (tensor - self.mean) / self.std

    def inverse(self) -> Callable:
        return lambda tensor: tensor * self.std + self.mean


class LogarithmTransform(ReversableTransform):
    def __init__(self, log_shift=0, final_shift=0):
        self.log_shift = 0
        self.final_shift = 0

    def __call__(self, tensor):
        return torch.log(tensor + self.log_shift) + self.final_shift

    def inverse(self):
        return lambda tensor: torch.exp(tensor - self.final_shift) - self.log_shift


__all__ = ['ReversableTransform', 'CustomNormalize', 'LogarithmTransform']