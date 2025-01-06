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

from trifast.utils import gen_tensors, enable_tf32, disable_tf32


warmup = 2
repeat = 10


rows = []

h = 4
d = 32

for n in [32, 64, 128, 256, 512, 1024, 2048]:
    for dtype in [torch.bfloat16, torch.float16, torch.float32]:
        for mode in ["fwd", "bwd"]:
            for name, model in [
                ("trifast", triangle_attention),
                ("simple", disable_tf32(triangle_attention_simple)),
                ("simple_tf32", enable_tf32(triangle_attention_simple)),
                ("simple_compiled", disable_tf32(torch.compile(triangle_attention_simple))),
                ("simple_compiled_tf32", enable_tf32(torch.compile(triangle_attention_simple))),
                ("ds4s", triangle_self_attention_ds4s),
            ]:
                torch.cuda.synchronize()
                print(f"{name} {n} {mode} {dtype}")

                for _ in range(warmup):
                    try:
                        q, k, v, bias, mask = gen_tensors(
                            n=n,
                            h=h,
                            d=d,
                            dtype=dtype,
                            device=torch.device("cuda"),
                            use_mask=True,
                        )
                    except torch.cuda.OutOfMemoryError:
                        break
                    try:
                        out = model(q, k, v, bias, mask)
                        torch.cuda.synchronize()
                        if mode == "bwd":
                            out.sum().backward()
                            torch.cuda.synchronize()
                    except torch.cuda.OutOfMemoryError:
                        print(f"OOM {name} {n} {mode}")

                for _ in range(repeat):
                    torch.cuda.reset_peak_memory_stats()

                    try:
                        q, k, v, bias, mask = gen_tensors(
                            n=n,
                            h=h,
                            d=d,
                            dtype=dtype,
                            device=torch.device("cuda"),
                            use_mask=True,
                        )
                    except torch.cuda.OutOfMemoryError:
                        break

                    try:
                        out = model(q, k, v, bias, mask)
                        torch.cuda.synchronize()
                        if mode == "bwd":
                            out.sum().backward()
                            torch.cuda.synchronize()
                    except torch.cuda.OutOfMemoryError:
                        print(f"OOM {name} {n} {mode}")
                        peak_memory = float("inf")
                    else:
                        peak_memory = (
                            torch.cuda.max_memory_allocated() / 1024**2
                        )  # convert to mb

                    rows.append(
                        {
                            "n": n,
                            "peak_memory": peak_memory,
                            "model": name,
                            "mode": mode,
                            "dtype": str(dtype),
                        }
                    )
                    torch.cuda.synchronize()

df = pd.DataFrame(rows)

out_dir = Path(__file__).parent.parent / "benchmark_plots"
out_dir.mkdir(exist_ok=True)
df.to_parquet(out_dir / "memory.parquet")

for mode in ["fwd", "bwd"]:
    for dtype in ["torch.float16", "torch.bfloat16", "torch.float32"]:
        f = df[df["mode"] == mode]
        f = f[f["dtype"] == dtype]
        sns.lineplot(data=f, x="n", y="peak_memory", hue="model")
        plt.title(f"Peak memory usage ({mode}), 3090, {dtype}")
        plt.savefig(out_dir / f"peak_memory_{mode}_{dtype}.png")
        plt.close()

pivot_df = (
    df.groupby(["n", "model", "mode", "dtype"])
    .mean()
    .reset_index()
    .pivot(index="n", columns=["model", "mode", "dtype"], values="peak_memory")
)


print(pivot_df)
