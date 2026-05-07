def has_cuda():
    import torch

    return torch.cuda.is_available()


def has_cuda_nvidia():
    import torch

    if not torch.cuda.is_available():
        return False
    if torch.version.cuda is None:
        return False
    try:
        name = torch.cuda.get_device_name(0)
    except Exception:
        return False
    return "nvidia" in name.lower()


def has_rocm():
    import torch

    if not torch.cuda.is_available():
        return False
    return hasattr(torch.version, "hip") and torch.version.hip is not None
