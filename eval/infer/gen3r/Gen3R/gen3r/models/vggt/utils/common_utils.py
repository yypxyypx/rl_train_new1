import torch
import torch.nn as nn
from typing import List, Tuple
from torch import Tensor


def get_parameter_dtype(parameter: torch.nn.Module) -> torch.dtype:
    """
    Returns the first found floating dtype in parameters if there is one, otherwise returns the last dtype it found.
    """
    # 1. Check if we have attached any dtype modifying hooks (eg. layerwise casting)
    if isinstance(parameter, nn.Module):
        for name, submodule in parameter.named_modules():
            if not hasattr(submodule, "_diffusers_hook"):
                continue
            registry = submodule._diffusers_hook
            hook = registry.get_hook("layerwise_casting")
            if hook is not None:
                return hook.compute_dtype

    # 2. If no dtype modifying hooks are attached, return the dtype of the first floating point parameter/buffer
    last_dtype = None

    for name, param in parameter.named_parameters():
        last_dtype = param.dtype
        if param.is_floating_point():
            return param.dtype

    for buffer in parameter.buffers():
        last_dtype = buffer.dtype
        if buffer.is_floating_point():
            return buffer.dtype

    if last_dtype is not None:
        # if no floating dtype was found return whatever the first dtype is
        return last_dtype

    # For nn.DataParallel compatibility in PyTorch > 1.5
    def find_tensor_attributes(module: nn.Module) -> List[Tuple[str, Tensor]]:
        tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
        return tuples

    gen = parameter._named_members(get_members_fn=find_tensor_attributes)
    last_tuple = None
    for tuple in gen:
        last_tuple = tuple
        if tuple[1].is_floating_point():
            return tuple[1].dtype

    if last_tuple is not None:
        # fallback to the last dtype
        return last_tuple[1].dtype