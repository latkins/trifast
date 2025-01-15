import os
from collections import defaultdict
import torch
import json
import triton
from functools import partial
from importlib.resources import files

# optional env var?
cache_dir = files("trifast") / "configs"


FORCE_RETUNE = os.getenv("TRIFAST_FORCE_RETUNE", "0").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
IS_TESTING = os.getenv("PYTEST_VERSION")

device_capability = torch.cuda.get_device_capability()
device_capability = f"{device_capability[0]}-{device_capability[1]}"

device_name = torch.cuda.get_device_name().replace(" ", "-")

# Allowed values -> should be compute cap dependant?
allowed = {
    "block_j": [16, 32, 64, 128, 256],
    "block_k": [16, 32, 64, 128, 256],
    "warps": [1, 2, 4, 8],
    "stages": [1, 2, 3, 4, 5, 6],
}


def config_to_dict(config: triton.Config) -> dict:
    # This assume we are not making use of `pre_hook` in the `triton.Config`
    return {
        "kwargs": config.kwargs,
        "num_warps": config.num_warps,
        "num_stages": config.num_stages,
        "num_ctas": config.num_ctas,
        "maxnreg": config.maxnreg,
    }


def dict_to_config(d: dict) -> triton.Config:
    return triton.Config(
        kwargs=d["kwargs"],
        num_warps=d["num_warps"],
        num_stages=d["num_stages"],
        num_ctas=d["num_ctas"],
        maxnreg=d["maxnreg"],
    )


def get_neighbors(current: int, allowed: list[int], n_neighbours: int = 1) -> list[int]:
    """Get the next lower and higher values from allowed list"""
    idx = allowed.index(current)
    return list(set(allowed[idx - n_neighbours : idx + n_neighbours + 1]))


# THIS IS COMICALLY BRITTLE. RELIES ON THE ORDER OF KEYS PASSED TO AUTOTUNE.
def parse_config_key(key: str) -> dict:
    h, dim, n, dtype, *_ = key.split("_")

    h = int(h)
    dim = int(dim)
    n = int(n)

    return {"h": h, "dim": dim, "N": n, "dtype": dtype}


def parse_config_file(json_data):
    # dtype -> h -> dim -> n -> config
    lookup = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for k, v in json_data.items():
        parsed = parse_config_key(k)
        lookup[parsed["dtype"]][parsed["h"]][parsed["dim"]][parsed["N"]] = (
            dict_to_config(v)
        )

    return lookup


def get_config_lookup(fn_name: str):
    file_path = cache_dir / f"{fn_name}_{device_name}_{device_capability}.json"

    if not file_path.exists():
        json_data = {}

    else:
        with file_path.open("r") as f:
            json_data = json.load(f)

    lookup = parse_config_file(json_data)

    return lookup


def get_nearest_config(
    lookup: dict, dtype: str, h: int, dim: int, n: int
) -> dict | None:
    try:
        config = lookup[dtype][h][dim][n]
    except KeyError:
        pass
    else:
        return config

    if dtype not in lookup:
        return None

    h_dict = lookup[dtype].get(h)
    if h_dict is None:
        # Fall back to nearest h
        h_values = sorted(lookup[dtype].keys())
        h_idx = None
        for idx, val in enumerate(h_values):
            if val >= h:
                h_idx = idx
                break
        if h_idx is None:
            return None
        h_dict = lookup[dtype][h_values[h_idx]]

    dim_dict = h_dict.get(dim)
    if dim_dict is None:
        # Fall back to nearest dim
        dim_values = sorted(h_dict.keys())
        dim_idx = None
        for idx, val in enumerate(dim_values):
            if val >= dim:
                dim_idx = idx
                break
        if dim_idx is None:
            return None
        dim_dict = h_dict[dim_values[dim_idx]]

    config = dim_dict.get(n)
    if config is not None:
        return config

    # Fall back to nearest n
    n_values = sorted(dim_dict.keys())
    n_idx = None
    for idx, val in enumerate(n_values):
        if val >= n:
            n_idx = idx
            break
    if n_idx is None:
        return None

    return dim_dict[n_values[n_idx]]


def block_valid(block: int, n: int, allowed: list[int]) -> bool:
    # n < 16, block > 16 invalid
    # n 128, block > 128 invalid
    # n > 256 all blocks valid

    if n <= allowed[0]:
        return block <= allowed[0]
    elif n >= allowed[-1]:
        return True

    return block < 2 * n


def block_idx_dist(block, comp_block, allowed):
    idx = allowed.index(block)
    comp_idx = allowed.index(comp_block)

    return abs(idx - comp_idx)


def prune_configs(configs, named_args, *, lookup, n_neighbours: int, **kwargs):
    q = named_args["q_ptr"]

    dtype = str(q.dtype)
    h, n, _, dim = q.shape

    config = lookup[dtype][h][dim].get(n, None)

    # We have an exact match, so just use that, unless we FORCE_RETUNE.
    if config is not None and not FORCE_RETUNE:
        return config

    # Find the closest matching configuration
    starting_config = get_nearest_config(lookup, dtype, h, dim, n)

    # If we don't have a starting config, apply some heuristics.
    if starting_config is None:
        # If we are < smallest allowed value, only allow smallest allowed block_j / block_k.
        pruned_configs = []
        for config in configs:
            smaller_than_min = n < allowed["block_j"][0] or n < allowed["block_k"][0]
            bigger_than_max = n > allowed["block_j"][-1] or n > allowed["block_k"][-1]
            # Don't allow block_j/block_k > 2x our size, unless we are already at the max, or are under the min..

            if not block_valid(config.kwargs["BLOCK_J"], n, allowed["block_j"]):
                continue

            if not block_valid(config.kwargs["BLOCK_K"], n, allowed["block_k"]):
                continue

        return configs

    pruned_configs = []
    # limit configs to those within n_neighbours of the starting config
    block_j = starting_config.kwargs["BLOCK_J"]
    block_k = starting_config.kwargs["BLOCK_K"]
    num_warps = starting_config.num_warps
    num_stages = starting_config.num_stages
    for config in configs:
        j = config.kwargs["BLOCK_J"]
        k = config.kwargs["BLOCK_K"]
        w = config.num_warps
        s = config.num_stages

        if block_idx_dist(block_j, j, allowed["block_j"]) > n_neighbours:
            continue
        if block_idx_dist(block_k, k, allowed["block_k"]) > n_neighbours:
            continue
        if abs(num_warps - config.num_warps) > n_neighbours:
            continue
        if abs(num_stages - config.num_stages) > n_neighbours:
            continue

        if not block_valid(config.kwargs["BLOCK_J"], n, allowed["block_j"]):
            continue
        if not block_valid(config.kwargs["BLOCK_K"], n, allowed["block_k"]):
            continue

        pruned_configs.append(config)

    return pruned_configs


fwd_lookup = get_config_lookup("_fwd")
bwd_kv_lookup = get_config_lookup("_bwd_kv")
bwd_q_lookup = get_config_lookup("_bwd_q")
bwd_b_lookup = get_config_lookup("_bwd_b")


prune_fwd = partial(prune_configs, lookup=fwd_lookup, n_neighbours=1)
prune_bwd_kv = partial(prune_configs, lookup=bwd_kv_lookup, n_neighbours=1)
prune_bwd_q = partial(prune_configs, lookup=bwd_q_lookup, n_neighbours=1)
prune_bwd_b = partial(prune_configs, lookup=bwd_b_lookup, n_neighbours=1)


safe_config = triton.Config(
    kwargs={"BLOCK_J": 16, "BLOCK_K": 16},
    num_warps=1,
    num_stages=1,
)

configs = (
    [
        triton.Config(
            kwargs={"BLOCK_J": block_j, "BLOCK_K": block_k},
            num_warps=warps,
            num_stages=stages,
        )
        for block_j in allowed["block_j"]
        for block_k in allowed["block_k"]
        for warps in allowed["warps"]
        for stages in allowed["stages"]
    ]
    if not IS_TESTING
    else [safe_config]
)
