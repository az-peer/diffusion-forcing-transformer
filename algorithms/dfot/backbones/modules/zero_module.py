from torch import nn
from einops import rearrange, parse_shape


def zero_module(module: nn.Module) -> nn.Module:
    """
    Zero out the parameters of a module and return it. This is the concept behind the idea
    that we want each b,ock of the DiT to be the identity block. Essentially this can
    help with stabililty. This is my understanding. We need to set some foundation
    for what type of activations a block of a crazy network might learn. If we didnt
    during the first pass our network might learn some crazy shit. This helps with
    inution. Start very slow and then slowly learn more complex things.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module
