import torch
from trifast.torch import _triangle_attention
from trifast.utils import gen_tensors
from torch.library import opcheck


def test_opcheck():
    for n in [16, 128, 256]:
        for d in [16, 32, 64]:
            for h in [1, 4]:
                q, k, v, b, m = gen_tensors(n, d, h, True, "cuda", dtype=torch.bfloat16)
                # opcheck raises an exception on failure
                opcheck(_triangle_attention, (q, k, v, b, m), raise_exception=True)
