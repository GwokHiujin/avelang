from abc import ABCMeta, abstractmethod


class DriverBase(metaclass=ABCMeta):
    @classmethod
    @abstractmethod
    def is_active(cls):
        pass

    @classmethod
    @abstractmethod
    def get_current_target(cls):
        pass

    def __init__(self) -> None:
        pass


class GPUDriver(DriverBase):
    def __init__(self):
        # TODO: support other frameworks than torch
        import torch

        self.get_device_capability = torch.cuda.get_device_capability
        try:
            from torch._C import _cuda_getCurrentRawStream

            self.get_current_stream = _cuda_getCurrentRawStream
        except ImportError:
            self.get_current_stream = lambda idx: torch.cuda.current_stream(idx).cuda_stream
        self.get_current_device = torch.cuda.current_device
        self.set_current_device = torch.cuda.set_device
