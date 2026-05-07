from abc import ABCMeta, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class GPUTarget:
    tuple: str
    chip: str


class BaseBackend(metaclass=ABCMeta):
    def __init__(self, target: GPUTarget):
        self.target = target

    @abstractmethod
    def parse_options(self, options: dict) -> object:
        """
        Converts an `options` dictionary into an arbitrary object and returns it.
        This function may contain target-specific heuristics and check the legality of the provided options
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def supports_target(target: GPUTarget):
        raise NotImplementedError

    @staticmethod
    def get_int_specialization(arg, **kwargs):
        return ""

    @staticmethod
    def get_tensor_specialization(arg, **kwargs):
        return ""
