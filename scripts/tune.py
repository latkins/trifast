import triton
import torch
from trifast.triton import (
    _fwd,
    _bwd_q,
    _bwd_kv,
    _bwd_b,
)
import random
from tqdm import tqdm
from pathlib import Path
import time
import pandas as pd
import yaml


device_cap = torch.cuda.get_device_capability()
device_name = torch.cuda.get_device_name()


def get_tensors(n: int, d: int, h: int, scale, device="cuda", dtype=torch.bfloat16):
    torch.manual_seed(0)
    q = torch.empty((h, n, n, d), device=device, dtype=dtype).normal_(
        mean=0.0, std=scale
    )
    k = torch.empty((h, n, n, d), device=device, dtype=dtype).normal_(
        mean=0.0, std=scale
    )
    v = torch.empty((h, n, n, d), device=device, dtype=dtype).normal_(
        mean=0.0, std=scale
    )
    bias = torch.empty((h, n, n), device=device, dtype=dtype).normal_(
        mean=0.0, std=scale
    )

    o = torch.randn((h, n, n, d), device=device, dtype=dtype)
    do = torch.randn((h, n, n, d), device=device, dtype=dtype)
    delta = torch.randn((h, n, n, d), device=device, dtype=dtype)

    l = torch.zeros((h, n, n), device=device, dtype=torch.float32)

    mask = torch.randint(0, 2, (n, n), device=device, dtype=dtype)
    return {
        "q": q,
        "k": k,
        "v": v,
        "b": bias,
        "mask": mask,
        "l": l,
        "o": o,
        "do": do,
        "d": delta,
    }


def gen_ns(min_n, max_n, num_samples=5):
    base_ns = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    base_ns = [b for b in base_ns if min_n <= b <= max_n]

    for i in range(1, len(base_ns)):
        a, b = base_ns[i - 1], base_ns[i]

        yield a

        delta = b - a
        vals = delta // num_samples

        for j in range(1, num_samples):
            yield a + j * vals

        yield b


def gen_data(
    n,
    heads: list[int],
    dims: list[int],
    dtype,
    device,
    scale,
):
    random.seed(0)
    for h in heads:
        for d in dims:
            try:
                yield get_tensors(n, d, h, scale, device, dtype)
            except torch.OutOfMemoryError:
                continue


def time_with_timeout(func, timeout_ms, *args, **kwargs):
    """
    Time a CUDA function with timeout, returning both GPU and wall time

    Args:
        func: Function to time
        *args: Arguments to pass to the function
        timeout_ms: Maximum allowed GPU execution time in milliseconds

    Returns:
        tuple: (cuda_time_ms, wall_time_ms) or (None, wall_time_ms) if timeout
    """
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        stream = torch.cuda.Stream()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        wall_start = time.time()

        with torch.cuda.stream(stream):
            start.record(stream)
            result = func(*args, **kwargs)
            end.record(stream)

        # Keep checking if the operation is done. This means we don't need to sync.
        while not stream.query():
            wall_time = (time.time() - wall_start) * 1000
            if wall_time > timeout_ms:
                return None, wall_time
            time.sleep(0.001)

        wall_time = (time.time() - wall_start) * 1000
        cuda_time = start.elapsed_time(end)

        return cuda_time, wall_time
    except Exception as e:
        print(e)
        return None, None


def run_forward(*, q, k, v, b, mask, BLOCK_J, BLOCK_K, num_warps, num_stages, **kwargs):
    sm_scale = q.shape[-1] ** -0.5
    # TODO: logic to flatten batch/head dims.
    h, m, n, dim = q.shape

    grid = lambda args: (triton.cdiv(n, BLOCK_J), n, h)

    o = torch.zeros_like(q)
    l = torch.zeros((h, n, n), device=q.device, dtype=torch.float32)

    # fmt: off
    kernel = _fwd.fn[grid](
        o, o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        l, l.stride(0), l.stride(1), l.stride(2),
        q, q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k, k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v, v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        b, b.stride(0), b.stride(1), b.stride(2),
        mask, mask.stride(0), mask.stride(1),
        neg_inf=torch.finfo(q.dtype).min,
        sm_scale=sm_scale, N=n, DIM=dim, H=h,
        BLOCK_J=BLOCK_J, BLOCK_K=BLOCK_K, num_warps=num_warps, num_stages=num_stages
    )


# Idea is to run each kernel at diff input sizes with different triton compile settings.
def run_dq(
    *, q, k, v, b, mask, l, do, d, BLOCK_J, BLOCK_K, num_warps, num_stages, **kwargs
):
    sm_scale = q.shape[-1] ** -0.5
    # TODO: logic to flatten batch/head dims.
    h, m, n, dim = q.shape

    grid = lambda args: (triton.cdiv(n, BLOCK_J), n, h)

    o = torch.zeros_like(q)
    l = torch.zeros((h, n, n), device=q.device, dtype=torch.float32)
    dq = torch.zeros_like(q)

    q_grid = lambda args: (triton.cdiv(n, BLOCK_J), n, h)
    # fmt: off
    _bwd_q.fn[q_grid](
        d, d.stride(0), d.stride(1), d.stride(2),
        q, q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k, k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v, v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        b, b.stride(0), b.stride(1), b.stride(2),
        l, l.stride(0), l.stride(1), l.stride(2),
        mask, mask.stride(0), mask.stride(1),
        do, do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        dq, dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
        sm_scale=sm_scale,
        neg_inf=torch.finfo(q.dtype).min,
        H=h, N=n, DIM=dim,
        BLOCK_J=BLOCK_J, BLOCK_K=BLOCK_K, num_warps=num_warps, num_stages=num_stages
    )


def run_dkv(
    *, d, q, k, v, b, l, mask, do, BLOCK_J, BLOCK_K, num_warps, num_stages, **kwargs
):
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    db = torch.zeros_like(b).to(torch.float32)
    dmask = torch.zeros_like(mask)

    sm_scale = q.shape[-1] ** -0.5

    h, m, n, dim = q.shape

    grid = lambda args: (triton.cdiv(n, BLOCK_K), n, h)
    # fmt: off
    _bwd_kv.fn[grid](
        d, d.stride(0), d.stride(1), d.stride(2),
        q, q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k, k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v, v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        b, b.stride(0), b.stride(1), b.stride(2),
        l, l.stride(0), l.stride(1), l.stride(2),
        mask, mask.stride(0), mask.stride(1),
        do, do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        dk, dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
        dv, dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
        sm_scale=sm_scale,
        neg_inf=torch.finfo(q.dtype).min,
        H=h, N=n, DIM=dim,
        BLOCK_J=BLOCK_J, BLOCK_K=BLOCK_K, num_warps=num_warps, num_stages=num_stages
    )


def run_db(
    *, d, q, k, v, b, l, mask, do, BLOCK_J, BLOCK_K, num_warps, num_stages, **kwargs
):
    db = torch.zeros_like(b).to(torch.float32)
    dmask = torch.zeros_like(mask)

    sm_scale = q.shape[-1] ** -0.5

    h, m, n, dim = q.shape

    grid = (triton.cdiv(n, BLOCK_J), triton.cdiv(n, BLOCK_K), h)
    # fmt: off
    _bwd_b.fn[grid](
        d, d.stride(0), d.stride(1), d.stride(2),
        q, q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k, k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v, v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        b, b.stride(0), b.stride(1), b.stride(2),
        l, l.stride(0), l.stride(1), l.stride(2),
        mask, mask.stride(0), mask.stride(1),
        do, do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        db, db.stride(0), db.stride(1), db.stride(2),
        sm_scale=sm_scale,
        neg_inf=torch.finfo(q.dtype).min,
        H=h, N=n, DIM=dim,
        BLOCK_J=BLOCK_J, BLOCK_K=BLOCK_K,
        num_warps=num_warps, num_stages=num_stages,
    )


def tune(reps, fn, name, root):
    blocks_j = [16, 32, 64, 128]
    blocks_k = [16, 32, 64, 128]
    warps = [1, 2, 4, 8]
    num_stages = [1, 2, 3, 4, 6]
    for dtype in [torch.float32, torch.bfloat16, torch.float16]:
        print(dtype)
        rows = []
        for n in tqdm(gen_ns(16, 1024, num_samples=1), desc="n"):
            # torch.cuda.empty_cache()
            for data in tqdm(
                gen_data(
                    n,
                    [1, 2, 4, 8],
                    [32, 64],
                    dtype,
                    "cuda",
                    1.0,
                ),
                leave=False,
                desc="data",
            ):
                q = data["q"]
                n, h, d = q.shape[1], q.shape[0], q.shape[-1]
                # just so it doesn't take too long, keep track of speed for each setting.
                durs = []
                for BLOCK_J in tqdm(blocks_j, leave=False, desc="block_j"):
                    if BLOCK_J >= (2 * n) and n > 32:
                        continue
                    for BLOCK_K in tqdm(blocks_k, leave=False, desc="block_k"):
                        if BLOCK_K >= (2 * n) and n > 32:
                            continue
                        for num_warps in tqdm(warps, leave=False, desc="warp"):
                            for num_stage in tqdm(
                                num_stages, leave=False, desc="num_stage"
                            ):
                                # make sure we have compiled
                                try:
                                    fn(
                                        **data,
                                        BLOCK_J=BLOCK_J,
                                        BLOCK_K=BLOCK_K,
                                        num_warps=num_warps,
                                        num_stages=num_stage,
                                    )
                                except triton.runtime.errors.OutOfResources:
                                    break
                                except Exception as e:
                                    print(e)
                                    continue

                                torch.cuda.synchronize()

                                if len(durs) > 0:
                                    timeout_ms = sorted(durs)[0]
                                else:
                                    timeout_ms = 1000000

                                for _ in range(reps):
                                    cuda_dur, wall_dur = time_with_timeout(
                                        fn,
                                        timeout_ms,
                                        **data,
                                        BLOCK_J=BLOCK_J,
                                        BLOCK_K=BLOCK_K,
                                        num_warps=num_warps,
                                        num_stages=num_stage,
                                    )

                                    if cuda_dur is None:
                                        continue

                                    durs.append(wall_dur)

                                    rows.append(
                                        {
                                            "BLOCK_J": BLOCK_J,
                                            "BLOCK_K": BLOCK_K,
                                            "num_warps": num_warps,
                                            "num_stages": num_stage,
                                            "n": q.shape[1],
                                            "h": q.shape[0],
                                            "d": q.shape[-1],
                                            "cuda_time": cuda_dur,
                                            "wall_time": wall_dur,
                                            "dtype": str(q.dtype),
                                            "device_name": device_name,
                                            "device_cap": device_cap,
                                            "method": name,
                                        }
                                    )
        df = pd.DataFrame(rows)
        df_path = (
            root
            / f"{device_cap[0]}_{device_cap[1]}"
            / str(dtype)
            / f"{device_name}_{dtype}_{name}.parquet"
        )
        df_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(str(df_path))

        cfg_lookup = create_config_lookup(df)

        cfg_dir = (
            root.parent / "configs" / f"{device_cap[0]}_{device_cap[1]}" / str(dtype)
        )
        cfg_dir.mkdir(exist_ok=True, parents=True)

        with (cfg_dir / f"{name}.yaml").open("w") as f:
            f.write(cfg_lookup)


def create_config_lookup(df: pd.DataFrame) -> str:
    """Create grid configuration from DataFrame."""
    # Get mean cuda_time for each configuration
    mean_times = (
        df.groupby(["n", "h", "d", "BLOCK_J", "BLOCK_K", "num_warps", "num_stages"])[
            "cuda_time"
        ]
        .mean()
        .reset_index()
    )

    # For each n,h,d combo, find the row with minimum cuda_time
    best_params = mean_times.loc[
        mean_times.groupby(["n", "h", "d"])["cuda_time"].idxmin()
    ]

    grid = {
        "grid_points": {
            "n": sorted(df["n"].unique().tolist()),
            "h": sorted(df["h"].unique().tolist()),
            "d": sorted(df["d"].unique().tolist()),
        },
        "settings": {},
    }

    # Store best parameters for each n,h,d point
    for _, row in best_params.iterrows():
        grid["settings"][f"{int(row['n'])},{int(row['h'])},{int(row['d'])}"] = {
            "BLOCK_J": int(row["BLOCK_J"]),
            "BLOCK_K": int(row["BLOCK_K"]),
            "num_warps": int(row["num_warps"]),
            "num_stages": int(row["num_stages"]),
        }

    return yaml.dump(grid, sort_keys=False, indent=2)


root = Path(__file__).parent.parent / "tune"


tune(reps=3, fn=run_dkv, name="_bwd_kv", root=root)

tune(reps=3, fn=run_db, name="_bwd_b", root=root)

tune(reps=3, fn=run_dq, name="_bwd_q", root=root)

tune(reps=3, fn=run_forward, name="_fwd", root=root)
