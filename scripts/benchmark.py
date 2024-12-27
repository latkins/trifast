from pathlib import Path
import triton
import torch
import triton.testing
from pairformer_kernel.torch import triangle_attention
from pairformer_kernel.equiv import (
    triangle_attention_simple,
    triangle_self_attention_ds4s,
)

configs = [
    triton.testing.Benchmark(
        x_names=["n"],  # Argument names to use as x-axis
        x_vals=[32, 64, 128, 256, 512, 768, 1024, 2048],  # Different values for n to benchmark
        line_arg="provider",  # Argument name whose value corresponds to different lines in the plot
        line_vals=["compiled", "simple", "kernel", "ds4s"],  # Values for the line_arg
        line_names=[
            "Compiled Simple Pytorch",
            "Simple Pytorch",
            "Triton Kernel",
            "Deepspeed",
        ],  # Labels for the lines
        styles=[("green", "--"), ("red", "--"), ("blue", "-"), ("orange", ":")],  # Line styles
        # line_vals=['kernel'],  # Values for the line_arg
        # line_names=['Triton Kernel'],  # Labels for the lines
        # styles=[('blue', '-')],  # Line styles
        ylabel="milliseconds",  # Label name for the y-axis
        plot_name=f"tri_attn_{mode}",
        args={"mode": mode},  # Other arguments to pass to the function
    )
    for mode in ["bwd", "fwd"]
]


@triton.testing.perf_report(configs)
def benchmark(n, mode, provider):
    assert mode in ["fwd", "bwd"]

    # this is what af3 uses for d and h.
    d = 32
    h = 4
    device = "cuda"
    dtype = torch.bfloat16

    quantiles = [0.5, 0.1, 0.9]
    warmup = 100
    rep = 200

    def get_tensors():
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

    print(provider, n, mode)
    q, k, v, bias, mask, do = get_tensors()

    if provider == "compiled":
        fn = lambda: torch.compile(triangle_attention_simple, fullgraph=True)(q, k, v, bias, mask)
    if provider == "simple":
        fn = lambda: triangle_attention_simple(q, k, v, bias, mask)
    if provider == "kernel":
        fn = lambda: triangle_attention(q, k, v, bias, mask)
    if provider == "ds4s":
        fn = lambda: triangle_self_attention_ds4s(q, k, v, bias, mask)

    try:
        if mode == "bwd":
            o = fn()
            fn = lambda: o.backward(do, retain_graph=True)

        ms, max_ms, min_ms = triton.testing.do_bench(
            fn,
            warmup=warmup,
            rep=rep,
            quantiles=quantiles,
        )
    except torch.cuda.OutOfMemoryError:
        ms, max_ms, min_ms = float("inf"), float("inf"), float("inf")

    return ms, max_ms, min_ms

out_dir = Path(__file__).parent.parent
out_dir.mkdir(exist_ok=True)
benchmark.run(
    print_data=True,
    show_plots=True,
    save_path=str(out_dir / "benchmark_plots"),
)
