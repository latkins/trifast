from pathlib import Path
import triton
import torch
import triton.testing
from trifast.torch import triangle_attention
from trifast.equiv import (
    triangle_attention_simple,
    triangle_self_attention_ds4s,
)
from trifast.utils import gen_tensors, disable_tf32, enable_tf32

configs = [
    triton.testing.Benchmark(
        x_names=["n"],  # Argument names to use as x-axis
        x_vals=[
            32,
            64,
            128,
            256,
            512,
            768,
            1024,
            2048,
        ],  # Different values for n to benchmark
        line_arg="provider",  # Argument name whose value corresponds to different lines in the plot
        line_vals=[
            "simple",
            "simple_tf32",
            "simple_compiled",
            "simple_compiled_tf32",
            "trifast",
            "ds4s",
        ],  # Values for the line_arg
        line_names=[
            "Simple Pytorch",
            "Simple Pytorch, TF32 enabled",
            "Simple Pytorch, Compiled",
            "Simple Pytorch, Compiled, TF32 enabled",
            "Trifast",
            "Deepspeed",
        ],  # Labels for the lines
        styles=[
            ("green", "-."),
            ("red", "-."),
            ("yellow", "-."),
            ("pink", "-."),
            ("blue", ":"),
            ("orange", ":"),
        ],  # Line styles
        ylabel="milliseconds",  # Label name for the y-axis
        plot_name=f"tri_attn_{mode}_{dtype}",
        args={"mode": mode, "dtype": dtype},  # Other arguments to pass to the function
    )
    for mode in ["bwd", "fwd"]
    for dtype in [torch.float32, torch.float16, torch.bfloat16]
]


@triton.testing.perf_report(configs)
def benchmark(n, mode, dtype, provider):
    assert mode in ["fwd", "bwd"]

    # this is what af3 uses for d and h.
    d = 32
    h = 4

    quantiles = [0.5, 0.1, 0.9]
    warmup = 5
    rep = 100

    print(provider, n, mode, dtype)
    q, k, v, bias, mask = gen_tensors(
        n=n,
        h=h,
        d=d,
        dtype=dtype,
        device=torch.device("cuda"),
        use_mask=True,
    )

    if provider == "simple":
        fn = lambda: disable_tf32(triangle_attention_simple)(q, k, v, bias, mask)
    if provider == "simple_tf32":
        fn = lambda: enable_tf32(triangle_attention_simple)(q, k, v, bias, mask)
    if provider == "simple_compiled":
        fn = lambda: disable_tf32(torch.compile(triangle_attention_simple))(
            q, k, v, bias, mask
        )
    if provider == "simple_compiled_tf32":
        fn = lambda: enable_tf32(torch.compile(triangle_attention_simple))(
            q, k, v, bias, mask
        )
    if provider == "trifast":
        fn = lambda: triangle_attention(q, k, v, bias, mask)
    if provider == "ds4s":
        fn = lambda: triangle_self_attention_ds4s(q, k, v, bias, mask)

    try:
        if mode == "bwd":
            o = fn()
            s = o.sum()
            fn = lambda: s.backward(retain_graph=True)

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
benchmark.run(
    print_data=True,
    show_plots=True,
    save_path=str(out_dir / "benchmark_plots"),
)
