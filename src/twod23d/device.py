"""The one place that decides where models run. Stages use ctx.device and never choose on their own."""


def pick_device(requested: str = "auto") -> str:
    """Return a torch device string. "auto" means CUDA if it is available, else CPU."""
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:  # no model stages installed, so nothing needs a GPU
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"
