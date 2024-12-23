import torch
from pairformer_kernel.torch import triangle_attention
from pairformer_kernel.equiv import (
    triangle_self_attention_ds4s,
    triangle_attention_simple,
)
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def get_tensors(n=128, d=32, h=2, device="cuda", dtype=torch.bfloat16):
    torch.manual_seed(0)
    q = torch.randn(h, n, n, d).to(device, dtype=dtype)
    k = torch.randn(h, n, n, d).to(device, dtype=dtype)
    v = torch.randn(h, n, n, d).to(device, dtype=dtype)
    bias = torch.randn(h, n, n).to(device, dtype=dtype)
    mask = torch.zeros(n, n).to(device) > 0
    return q, k, v, bias, mask


repeat = 10
warmup = 50


rows = []

for n in [32, 64, 128, 256, 512, 1024, 2048]:
    print(f"Doing {n}")
    for name, model in [
        ("simple", triangle_attention_simple),
        ("triton", triangle_attention),
        ("ds4s", triangle_self_attention_ds4s),
    ]:
        if name == "simple" and n > 512:
            continue

        for _ in range(warmup):
            q, k, v, bias, mask = get_tensors(n=n)
            out = triangle_attention(q, k, v, bias, mask)

        for _ in range(repeat):
            torch.cuda.reset_peak_memory_stats()

            q, k, v, bias, mask = get_tensors(n=n)
            out = model(q, k, v, bias, mask)

            peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # convert to mb

            rows.append({"n": n, "peak_memory": peak_memory, "model": name})

df = pd.DataFrame(rows)

pivot_df = (
    df.groupby(["n", "model"])
    .mean()
    .reset_index()
    .pivot(index="n", columns="model", values="peak_memory")
)


print(pivot_df)

sns.lineplot(data=df, x="n", y="peak_memory", hue="model")
plt.show()
