import os
from collections import defaultdict
import torch
import json
import triton
from functools import partial
from importlib.resources import files

# optional env var?
cache_dir = files("trifast") / "configs"


FORCE_TUNE = os.getenv("TRIFAST_FORCE_TUNE", "0").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
IS_TESTING = os.getenv("PYTEST_VERSION")

device_capability = torch.cuda.get_device_capability()
device_capability = f"{device_capability[0]}-{device_capability[1]}"

device_name = torch.cuda.get_device_name().replace(" ", "-")


def config_to_dict(config: triton.Config) -> dict:
    # This assume we are not making use of `pre_hook` in the `triton.Config`
    return {
        "kwargs": config.kwargs,
        "num_warps": config.num_warps,
        "num_stages": config.num_stages,
    }


def dict_to_config(d: dict) -> triton.Config:
    return triton.Config(
        kwargs=d["kwargs"],
        num_warps=d["num_warps"],
        num_stages=d["num_stages"],
    )


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


def prune_configs(configs, named_args, *, lookup, **kwargs):
    q = named_args["q_ptr"]

    dtype = str(q.dtype)
    h, n, _, dim = q.shape

    config = lookup[dtype][h][dim].get(n, None)

    # We have an exact match, so just use that, unless we FORCE_TUNE.
    if config is not None and not FORCE_TUNE:
        return config

    return configs


fwd_lookup = get_config_lookup("_fwd")
bwd_kv_lookup = get_config_lookup("_bwd_kv")
bwd_q_lookup = get_config_lookup("_bwd_q")
bwd_b_lookup = get_config_lookup("_bwd_b")


prune_fwd = partial(prune_configs, lookup=fwd_lookup)
prune_bwd_kv = partial(prune_configs, lookup=bwd_kv_lookup)
prune_bwd_q = partial(prune_configs, lookup=bwd_q_lookup)
prune_bwd_b = partial(prune_configs, lookup=bwd_b_lookup)

# Base configs that should be ~ok for things <= 512.
_fwd_configs = [
    triton.Config(kwargs={"BLOCK_J": 16, "BLOCK_K": 32}, num_warps=1, num_stages=2),
    triton.Config(kwargs={"BLOCK_J": 32, "BLOCK_K": 16}, num_warps=1, num_stages=2),
    triton.Config(kwargs={"BLOCK_J": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
    triton.Config(kwargs={"BLOCK_J": 64, "BLOCK_K": 16}, num_warps=2, num_stages=3),
    triton.Config(kwargs={"BLOCK_J": 32, "BLOCK_K": 32}, num_warps=1, num_stages=1),
    triton.Config(kwargs={"BLOCK_J": 128, "BLOCK_K": 16}, num_warps=2, num_stages=2),
    triton.Config(kwargs={"BLOCK_J": 128, "BLOCK_K": 32}, num_warps=2, num_stages=2),
    triton.Config(kwargs={"BLOCK_J": 32, "BLOCK_K": 64}, num_warps=2, num_stages=2),
    triton.Config(kwargs={"BLOCK_J": 64, "BLOCK_K": 32}, num_warps=2, num_stages=3),
    triton.Config(kwargs={"BLOCK_J": 32, "BLOCK_K": 16}, num_warps=1, num_stages=4),
]
if FORCE_TUNE:
    _fwd_configs.extend(
        [
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 32}, num_warps=1, num_stages=5),
            triton.Config({"BLOCK_J": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_J": 128, "BLOCK_K": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 64}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_J": 64, "BLOCK_K": 16}, num_warps=2, num_stages=3),
            triton.Config({"BLOCK_J": 128, "BLOCK_K": 16}, num_warps=2, num_stages=2),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 16}, num_warps=2, num_stages=4),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 16}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 64}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK_J": 128, "BLOCK_K": 16}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 32}, num_warps=8, num_stages=1),
            triton.Config({"BLOCK_J": 64, "BLOCK_K": 32}, num_warps=8, num_stages=1),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 16}, num_warps=2, num_stages=5),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 16}, num_warps=8, num_stages=1),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 64}, num_warps=8, num_stages=1),
            triton.Config({"BLOCK_J": 64, "BLOCK_K": 64}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK_J": 128, "BLOCK_K": 32}, num_warps=8, num_stages=1),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 32}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 128}, num_warps=4, num_stages=2),
        ]
    )

_bwd_q_configs = [
    triton.Config({"BLOCK_J": 16, "BLOCK_K": 32}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_J": 32, "BLOCK_K": 16}, num_warps=1, num_stages=2),
    triton.Config({"BLOCK_J": 32, "BLOCK_K": 64}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_J": 32, "BLOCK_K": 32}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK_J": 64, "BLOCK_K": 16}, num_warps=2, num_stages=3),
    triton.Config({"BLOCK_J": 16, "BLOCK_K": 16}, num_warps=1, num_stages=2),
    triton.Config({"BLOCK_J": 32, "BLOCK_K": 16}, num_warps=1, num_stages=3),
    triton.Config({"BLOCK_J": 16, "BLOCK_K": 64}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK_J": 32, "BLOCK_K": 64}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_J": 64, "BLOCK_K": 32}, num_warps=2, num_stages=2),
]
if FORCE_TUNE:
    _bwd_q_configs.extend(
        [
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 128}, num_warps=2, num_stages=2),
            triton.Config({"BLOCK_J": 64, "BLOCK_K": 16}, num_warps=2, num_stages=3),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 64}, num_warps=1, num_stages=1),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 16}, num_warps=1, num_stages=3),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 32}, num_warps=2, num_stages=3),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 128}, num_warps=2, num_stages=3),
            triton.Config({"BLOCK_J": 128, "BLOCK_K": 16}, num_warps=2, num_stages=2),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 16}, num_warps=1, num_stages=4),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 64}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK_J": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_J": 128, "BLOCK_K": 32}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 32}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 32}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_J": 64, "BLOCK_K": 16}, num_warps=1, num_stages=2),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 64}, num_warps=2, num_stages=3),
            triton.Config({"BLOCK_J": 128, "BLOCK_K": 16}, num_warps=1, num_stages=2),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 32}, num_warps=1, num_stages=3),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 128}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK_J": 64, "BLOCK_K": 32}, num_warps=1, num_stages=2),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 128}, num_warps=4, num_stages=2),
        ]
    )


_bwd_kv_configs = [
    triton.Config({"BLOCK_J": 32, "BLOCK_K": 32}, num_warps=1, num_stages=2),
    triton.Config({"BLOCK_J": 32, "BLOCK_K": 16}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_J": 64, "BLOCK_K": 16}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_J": 32, "BLOCK_K": 32}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_J": 32, "BLOCK_K": 32}, num_warps=1, num_stages=3),
    triton.Config({"BLOCK_J": 64, "BLOCK_K": 32}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_J": 16, "BLOCK_K": 16}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_J": 32, "BLOCK_K": 16}, num_warps=1, num_stages=2),
    triton.Config({"BLOCK_J": 32, "BLOCK_K": 32}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK_J": 16, "BLOCK_K": 32}, num_warps=2, num_stages=1),
]

if FORCE_TUNE:
    _bwd_kv_configs.extend(
        [
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 128}, num_warps=2, num_stages=2),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 32}, num_warps=1, num_stages=4),
            triton.Config({"BLOCK_J": 64, "BLOCK_K": 16}, num_warps=2, num_stages=3),
            triton.Config({"BLOCK_J": 128, "BLOCK_K": 16}, num_warps=2, num_stages=3),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 64}, num_warps=2, num_stages=3),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 64}, num_warps=1, num_stages=1),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 16}, num_warps=1, num_stages=3),
            triton.Config({"BLOCK_J": 64, "BLOCK_K": 32}, num_warps=2, num_stages=4),
            triton.Config({"BLOCK_J": 64, "BLOCK_K": 64}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK_J": 128, "BLOCK_K": 32}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 128}, num_warps=2, num_stages=1),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 32}, num_warps=1, num_stages=3),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 64}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK_J": 64, "BLOCK_K": 16}, num_warps=1, num_stages=2),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 64}, num_warps=2, num_stages=4),
            triton.Config({"BLOCK_J": 128, "BLOCK_K": 16}, num_warps=1, num_stages=2),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 128}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK_J": 64, "BLOCK_K": 32}, num_warps=1, num_stages=2),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 16}, num_warps=4, num_stages=2),
        ]
    )

_bwd_b_configs = [
    triton.Config({"BLOCK_J": 16, "BLOCK_K": 16}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_J": 16, "BLOCK_K": 16}, num_warps=2, num_stages=3),
    triton.Config({"BLOCK_J": 16, "BLOCK_K": 32}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_J": 32, "BLOCK_K": 32}, num_warps=1, num_stages=2),
    triton.Config({"BLOCK_J": 32, "BLOCK_K": 64}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_J": 16, "BLOCK_K": 64}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_J": 16, "BLOCK_K": 16}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_J": 32, "BLOCK_K": 32}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_J": 16, "BLOCK_K": 16}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_J": 32, "BLOCK_K": 32}, num_warps=2, num_stages=3),
]

if FORCE_TUNE:
    _bwd_b_configs.extend(
        [
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 64}, num_warps=2, num_stages=4),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 64}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 32}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 32}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 16}, num_warps=2, num_stages=6),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 16}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 128}, num_warps=2, num_stages=4),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 64}, num_warps=2, num_stages=4),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 32}, num_warps=2, num_stages=5),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 16}, num_warps=2, num_stages=6),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 64}, num_warps=1, num_stages=2),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 32}, num_warps=2, num_stages=5),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 16}, num_warps=1, num_stages=6),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 64}, num_warps=1, num_stages=3),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 32}, num_warps=1, num_stages=4),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 64}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 16}, num_warps=1, num_stages=4),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 128}, num_warps=1, num_stages=3),
            triton.Config({"BLOCK_J": 32, "BLOCK_K": 32}, num_warps=1, num_stages=6),
            triton.Config({"BLOCK_J": 16, "BLOCK_K": 16}, num_warps=4, num_stages=4),
        ]
    )
