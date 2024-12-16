from torch import float32, float64, float16


def get_dtype_from_str(dtype_str: str):
    if dtype_str == "float32":
        return float32
    elif dtype_str == "float64":
        return float64
    elif dtype_str == "float16":
        return float16
    else:
        raise ValueError(
            f"Unsupported data type \"{dtype_str}\". Supported data types are: float32, float64, float16"
        )


def get_model_size(model):
    trainables, total = 0, 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainables += p.numel()
    return trainables, total


__all__ = ['get_model_size', 'get_dtype_from_str']