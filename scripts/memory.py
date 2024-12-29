import torch
from trifast.torch import triangle_attention
from trifast.equiv import (
    triangle_self_attention_ds4s,
    triangle_attention_simple,
)
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def get_tensors(n=128, d=32, h=2, device="cuda", dtype=torch.bfloat16, mode="fwd"):
    torch.manual_seed(0)
    q = torch.randn(
        (h, n, n, d), device=device, dtype=dtype, requires_grad=mode == "bwd"
    )
    k = torch.randn(
        (h, n, n, d), device=device, dtype=dtype, requires_grad=mode == "bwd"
    )
    v = torch.randn(
        (h, n, n, d), device=device, dtype=dtype, requires_grad=mode == "bwd"
    )

    bias = torch.randn(
        (h, n, n), device=device, dtype=dtype, requires_grad=mode == "bwd"
    )
    mask = torch.randn((n, n), device=device) > 0
    mask = torch.zeros_like(mask)

    do = torch.randn_like(q)

    return q, k, v, bias, mask, do


repeat = 10
warmup = 50


rows = []

for n in [32, 64, 128, 256, 512, 1024]:
    for mode in ["fwd", "bwd"]:
        for name, model in [
            ("simple", triangle_attention_simple),
            ("triton", triangle_attention),
            ("ds4s", triangle_self_attention_ds4s),
        ]:
            print(f"{name} {n} {mode}")

            for _ in range(warmup):
                q, k, v, bias, mask, do = get_tensors(n=n, mode=mode)
                try:
                    out = model(q, k, v, bias, mask)

                    if mode == "bwd":
                        out.backward(do)
                except torch.cuda.OutOfMemoryError:
                    print(f"OOM {name} {n} {mode}")

            for _ in range(repeat):
                torch.cuda.reset_peak_memory_stats()

                q, k, v, bias, mask, do = get_tensors(n=n, mode=mode)
                try:
                    out = model(q, k, v, bias, mask)

                    if mode == "bwd":
                        out.backward(do)
                except torch.cuda.OutOfMemoryError:
                    print(f"OOM {name} {n} {mode}")
                    peak_memory = float("inf")
                else:
                    peak_memory = (
                        torch.cuda.max_memory_allocated() / 1024**2
                    )  # convert to mb

                rows.append(
                    {"n": n, "peak_memory": peak_memory, "model": name, "mode": mode}
                )

df = pd.DataFrame(rows)


out_dir = Path(__file__).parent.parent / "benchmark_plots"
out_dir.mkdir(exist_ok=True)
for mode in ["fwd", "bwd"]:
    f = df[df["mode"] == mode]
    sns.lineplot(data=f, x="n", y="peak_memory", hue="model")
    plt.title(f"Peak memory usage ({mode}), 3090, bfloat16")
    plt.savefig(f"peak_memory_{mode}.png")
    plt.close()

pivot_df = (
    df.groupby(["n", "model", "mode"])
    .mean()
    .reset_index()
    .pivot(index="n", columns=["model", "mode"], values="peak_memory")
)

print(pivot_df)
