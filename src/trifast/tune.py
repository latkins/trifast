from math import log2
import os

os.environ["TRIFAST_FORCE_TUNE"] = "1"
os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

import torch
import argparse
from trifast.torch import triangle_attention
from trifast.utils import gen_tensors


def valid_dtypes(value: str) -> list[torch.dtype]:
    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    allowed_dtypes = set(dtype_map.keys()) | {"all"}

    dtypes = {dtype.strip().lower() for dtype in value.split(",")}

    invalid_dtypes = dtypes - allowed_dtypes
    if invalid_dtypes:
        raise argparse.ArgumentTypeError(
            f"Invalid dtype(s): {invalid_dtypes}. "
            f"Allowed values are: {allowed_dtypes}"
        )

    if "all" in dtypes:
        return list(dtype_map.values())

    return [dtype_map[dtype] for dtype in dtypes]


def parse_int_list(value: str) -> list[int]:
    try:
        return [int(x.strip()) for x in value.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(
            'Invalid format. Please provide comma-separated integers (e.g., "256,512,1024")'
        )


def tune(min_n, max_n, heads, dims, dtypes):
    ns = [2**i for i in range(int(log2(min_n)), int(log2(max_n)) + 1)]
    for d in dims:
        for h in heads:
            for n in ns:
                for dtype in dtypes:
                    print(f"Tuning: N: {n}, H: {h}, D: {d}, dtype: {dtype}")
                    q, k, v, bias, mask = gen_tensors(
                        n,
                        d,
                        h,
                        use_mask=True,
                        device=torch.device("cuda"),
                        dtype=dtype,
                        std=1.0,
                    )

                    tri_out = triangle_attention(q, k, v, bias, mask)
                    tri_out.sum().backward()


def main():
    parser = argparse.ArgumentParser(
        description="Autotune TriFast kernels over these parameters."
    )

    parser.add_argument(
        "--min-n", type=int, help="Miniumum sequence length to tune.", default=16
    )

    parser.add_argument(
        "--max-n", type=int, help="Maximum sequence length to tune.", default=1024
    )

    parser.add_argument(
        "--dtype",
        type=valid_dtypes,
        help="PyTorch datatypes to use (comma-separated). Options: float32, bfloat16, float16, all",
        default=[torch.bfloat16],
    )

    parser.add_argument(
        "--h",
        type=parse_int_list,
        help='List of number of heads (comma-separated integers, e.g., "1,2,4")',
        default=[4],
    )

    parser.add_argument(
        "--d",
        type=parse_int_list,
        help='List of dims (comma-separated integers, e.g., "16,32,64")',
        default=[32],
    )

    args = parser.parse_args()

    tune(args.min_n, args.max_n, args.h, args.d, args.dtype)


if __name__ == "__main__":
    main()
