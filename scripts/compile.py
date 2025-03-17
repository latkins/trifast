import torch
from trifast import triangle_attention
from trifast.utils import gen_tensors

torch._dynamo.config.compiled_autograd = True

device = torch.device('cuda')
dtype = torch.bfloat16
scale = 1.0

n = 256
d = 32
h = 4
q, k, v, bias, mask = gen_tensors(n, d, h, use_mask=True, device=device, dtype=dtype, std=scale, batch=3)

explanation = torch._dynamo.explain(triangle_attention, q, k, v, bias, mask)
print(explanation)

cmp = torch.compile(triangle_attention, fullgraph=True, mode="reduce-overhead")

a = cmp(q, k, v, bias, mask)


a.sum().backward()




