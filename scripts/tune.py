import triton
import torch
from pairformer_kernel.triangle_kernel import (
    _fwd,
    _bwd_q_kernel,
    _bwd_kvb_kernel,
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
    base_ns = [32, 64, 128, 256, 512, 1024, 2048, 4096]
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
            yield get_tensors(n, d, h, scale, device, dtype)


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


def run_forward(*, q, k, v, b, mask, BLOCK_J, BLOCK_K, WARP, NUM_STAGES, **kwargs):
    sm_scale = q.shape[-1] ** -0.5
    # TODO: logic to flatten batch/head dims.
    h, m, n, dim = q.shape

    grid = lambda args: (triton.cdiv(n, BLOCK_J), n, h)

    o = torch.zeros_like(q)
    l = torch.zeros((h, n, n), device=q.device, dtype=torch.float32)

    # fmt: off
    kernel = _fwd[grid](
        o, o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        l, l.stride(0), l.stride(1), l.stride(2),
        q, q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k, k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v, v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        b, b.stride(0), b.stride(1), b.stride(2),
        mask, mask.stride(0), mask.stride(1),
        neg_inf=torch.finfo(q.dtype).min,
        sm_scale=sm_scale, N=n, DIM=dim,
        BLOCK_J=BLOCK_J, BLOCK_K=BLOCK_K, num_warps=WARP, num_stages=NUM_STAGES
    )


# Idea is to run each kernel at diff input sizes with different triton compile settings.
def run_dq(*, q, k, v, b, mask, l, do, d, BLOCK_J, BLOCK_K, WARP, NUM_STAGES, **kwargs):
    sm_scale = q.shape[-1] ** -0.5
    # TODO: logic to flatten batch/head dims.
    h, m, n, dim = q.shape

    grid = lambda args: (triton.cdiv(n, BLOCK_J), n, h)

    o = torch.zeros_like(q)
    l = torch.zeros((h, n, n), device=q.device, dtype=torch.float32)
    dq = torch.zeros_like(q)

    q_grid = lambda args: (triton.cdiv(n, BLOCK_J), n, h)
    # fmt: off
    _bwd_q_kernel[q_grid](
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
        H=h, M=n, N=n, DIM=dim,
        BLOCK_J=BLOCK_J, BLOCK_K=BLOCK_K, num_warps=WARP, num_stages=NUM_STAGES
    )


def run_dkvb(
    *, d, q, k, v, b, l, mask, do, BLOCK_J, BLOCK_K, WARP, NUM_STAGES, **kwargs
):
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    db = torch.zeros_like(b).to(torch.float32)
    dmask = torch.zeros_like(mask)

    sm_scale = q.shape[-1] ** -0.5

    h, m, n, dim = q.shape

    grid = lambda args: (triton.cdiv(n, BLOCK_K), n, h)
    # fmt: off
    _bwd_kvb_kernel[grid](
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
        db, db.stride(0), db.stride(1), db.stride(2),
        sm_scale=sm_scale,
        neg_inf=torch.finfo(q.dtype).min,
        H=h, M=n, N=n, DIM=dim,
        BLOCK_J=BLOCK_J, BLOCK_K=BLOCK_K, num_warps=WARP, num_stages=NUM_STAGES
    )


def tune(reps, fn, name, root):
    blocks_j = [16, 32, 64, 128]
    blocks_k = [16, 32, 64, 128]
    warps = [1, 2, 4, 8]
    num_stages = [1, 2, 3, 4, 6]
    for n in tqdm(gen_ns(16, 1024, num_samples=1), desc="n"):
        rows = []
        for data in tqdm(
            gen_data(
                n,
                [1, 2, 4, 8],
                [32, 64],
                torch.bfloat16,
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
            for block_j in tqdm(blocks_j, leave=False, desc="block_j"):
                for block_k in tqdm(blocks_k, leave=False, desc="block_k"):
                    for warp in tqdm(warps, leave=False, desc="warp"):
                        for num_stage in tqdm(
                            num_stages, leave=False, desc="num_stage"
                        ):
                            # make sure we have compiled
                            try:
                                fn(
                                    **data,
                                    BLOCK_J=block_j,
                                    BLOCK_K=block_k,
                                    WARP=warp,
                                    NUM_STAGES=num_stage,
                                )
                            except triton.runtime.errors.OutOfResources as e:
                                print(e)
                                break
                            except Exception as e:
                                print(e)
                                continue

                            torch.cuda.synchronize()
                            start_event = torch.cuda.Event(enable_timing=True)
                            end_event = torch.cuda.Event(enable_timing=True)

                            if len(durs) > 0:
                                timeout_ms = sorted(durs)[0]
                            else:
                                timeout_ms = 1000000

                            stream = torch.cuda.Stream()
                            for _ in range(reps):
                                start_event.record(stream)
                                cuda_dur, wall_dur = time_with_timeout(
                                    fn,
                                    timeout_ms,
                                    **data,
                                    BLOCK_J=block_j,
                                    BLOCK_K=block_k,
                                    WARP=warp,
                                    NUM_STAGES=num_stage,
                                )

                                if cuda_dur is None:
                                    continue

                                durs.append(wall_dur)

                                rows.append(
                                    {
                                        "block_j": block_j,
                                        "block_k": block_k,
                                        "warp": warp,
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
            root = Path(__file__).parent.parent / "tune"
            root.mkdir(exist_ok=True)
            path = root / f"{device_name}_{n}_{h}_{d}_{name}.parquet"

            df.to_parquet(str(path))

def create_config_lookup(df: pd.DataFrame) -> str:
    """Create grid configuration from DataFrame."""
    # Get mean cuda_time for each configuration
    mean_times = (
        df.groupby(["n", "h", "d", "block_j", "block_k", "warp", "num_stages"])[
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
            "block_j": int(row["block_j"]),
            "block_k": int(row["block_k"]),
            "warp": int(row["warp"]),
            "num_stages": int(row["num_stages"]),
        }

    return yaml.dump(grid, sort_keys=False, indent=2)


root = Path(__file__).parent.parent / "tune"
root.mkdir(exist_ok=True)
cfg_dir = root.parent / "configs"
cfg_dir.mkdir(exist_ok=True)

tune(reps=5, fn=run_forward, name="fwd", root=root)
tune(reps=5, fn=run_dq, name="bwd_dq", root=root)
tune(reps=5, fn=run_dkvb, name="bwd_dkvb", root=root)

df = pd.read_parquet(root)

fwd_yaml = create_config_lookup(df[df["method"] == "fwd"])

with open(cfg_dir / "fwd.yaml", "w") as f:
    f.write(fwd_yaml)

bwd_dq_yaml = create_config_lookup(df[df["method"] == "bwd_dq"])
with open(cfg_dir / "bwd_dq.yaml", "w") as f:
    f.write(bwd_dq_yaml)

bwd_dkvb_yaml = create_config_lookup(df[df["method"] == "bwd_dkvb"])
with open(cfg_dir / "bwd_dkvb.yaml", "w") as f:
    f.write(bwd_dkvb_yaml)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

