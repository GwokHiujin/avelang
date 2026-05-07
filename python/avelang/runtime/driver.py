from __future__ import annotations
from ..backends import backends, DriverBase


def _create_driver() -> DriverBase:
    available_drivers = [x.driver for x in backends.values() if x.driver.is_active()]

    if not available_drivers:
        raise RuntimeError("No GPU drivers available. Please install CUDA or ROCm/HIP.")

    try:
        import torch

        if torch.cuda.is_available():
            if hasattr(torch.version, "hip") and torch.version.hip is not None:
                for driver_class in available_drivers:
                    if "amdgpu" in driver_class.__name__.lower():
                        return driver_class()
            else:
                for driver_class in available_drivers:
                    if "nvidia" in driver_class.__name__.lower():
                        return driver_class()
    except Exception as e:
        raise RuntimeError(f"Error detecting GPU backend with PyTorch: {e}")


class DriverConfig:
    def __init__(self) -> None:
        self._default: DriverBase | None = None
        self._active: DriverBase | None = None

    @property
    def default(self) -> DriverBase:
        if self._default is None:
            self._default = _create_driver()
        return self._default

    @property
    def active(self) -> DriverBase:
        if self._active is None:
            self._active = self.default
        return self._active

    def set_active(self, driver: DriverBase) -> None:
        self._active = driver

    def reset_active(self) -> None:
        self._active = self.default


driver = DriverConfig()
