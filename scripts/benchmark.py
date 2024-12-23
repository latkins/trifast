import triton
import torch
import triton.testing
from pairformer_kernel.torch import triangle_attention
from pairformer_kernel.equiv import (
    triangle_attention_simple,
    triangle_self_attention_ds4s,
)

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["n"],  # Argument names to use as x-axis
        x_vals=[32, 64, 128, 256, 512, 1024],  # Different values for n to benchmark
        line_arg="provider",  # Argument name whose value corresponds to different lines in the plot
        line_vals=["simple", "kernel", "ds4s"],  # Values for the line_arg
        line_names=[
            "Simple Pytorch",
            "Triton Kernel",
            "Deepspeed",
        ],  # Labels for the lines
        styles=[("red", "--"), ("blue", "-"), ("orange", ":")],  # Line styles
        # line_vals=['kernel'],  # Values for the line_arg
        # line_names=['Triton Kernel'],  # Labels for the lines
        # styles=[('blue', '-')],  # Line styles
        ylabel="milliseconds",  # Label name for the y-axis
        plot_name="triangle-attention-performance",  # Name for the plot
        args={},  # Other arguments to pass to the function
    )
)
def benchmark(n, provider):
    d = 32
    h = 4
    device = "cuda"
    dtype = torch.bfloat16  # Can be changed to bfloat16 as needed

    quantiles = [0.5, 0.1, 0.9]
    warmup = 100
    rep = 200

    def get_tensors():
        torch.manual_seed(0)
        q = torch.randn(h, n, n, d).to(device, dtype=dtype)
        k = torch.randn(h, n, n, d).to(device, dtype=dtype)
        v = torch.randn(h, n, n, d).to(device, dtype=dtype)
        bias = torch.randn(h, n, n).to(device, dtype=dtype)
        mask = torch.randn(n, n).to(device) > 0
        return q, k, v, bias, mask

    print(provider, n)
    with torch.no_grad():
        if provider == "simple":
            if n < 1024:
                q, k, v, bias, mask = get_tensors()

                ms, max_ms, min_ms = triton.testing.do_bench(
                    lambda: triangle_attention_simple(q, k, v, bias, mask),
                    warmup=warmup,
                    rep=rep,
                    quantiles=quantiles,
                )
            else:
                ms, max_ms, min_ms = None, None, None

        if provider == "kernel":
            q, k, v, bias, mask = get_tensors()

            ms, max_ms, min_ms = triton.testing.do_bench(
                lambda: triangle_attention(q, k, v, bias, mask),
                warmup=warmup,
                rep=rep,
                quantiles=quantiles,
            )

        if provider == "ds4s":
            q, k, v, bias, mask = get_tensors()

            ms, max_ms, min_ms = triton.testing.do_bench(
                lambda: triangle_self_attention_ds4s(q, k, v, bias, mask),
                warmup=warmup,
                rep=rep,
                quantiles=quantiles,
            )

    return ms, max_ms, min_ms


benchmark.run(print_data=True, show_plots=True)
