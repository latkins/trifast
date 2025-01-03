from pathlib import Path
import triton
import yaml
from bisect import bisect_right
import torch
import logging
from triton.runtime.jit import KernelInterface

logger = logging.getLogger(__name__)


class ConfigLookup(KernelInterface):
    # TODO: This should probably also generate the tuned kernels?
    def __init__(self, fn, key_args, cfg_dir: Path):
        self.fn = fn
        self.param_lookup = ParamLookup.from_folder(cfg_dir, f"{fn.__name__}.yaml")
        self.key_args = key_args
        self.cache = {}

    def run(self, *args, **kwargs):
        key_values = []
        arg_dict = {**dict(zip(self.fn.arg_names, args)), **kwargs}

        dtype = None
        for key in self.key_args:
            if key in arg_dict:
                val = arg_dict[key]
                if hasattr(val, "dtype"):
                    key_values.append(str(val.dtype))
                    dtype = val.dtype
                else:
                    key_values.append(val)
        cache_key = tuple(key_values)

        if cache_key not in self.cache:
            n = arg_dict["N"]
            h = arg_dict["H"]
            d = arg_dict["DIM"]
            config = self.param_lookup[dtype].get_parameters(n, h, d)
            self.cache[cache_key] = config

        while True:
            try:
                config = self.cache[cache_key]
                result = self.fn.run(*args, **kwargs, **config)
                return result
            except triton.OutOfResources:
                new_stages = max(1, config.get("num_stages", 2) - 1)
                self.cache[cache_key] = {**config, "num_stages": new_stages}
                if new_stages == 1:
                    raise


def tuned_lookup(key, config_dir):
    def decorator(fn):
        return ConfigLookup(fn, key, config_dir)

    return decorator


class DtypeParamLookup:
    """Param lookup for a specific dtype."""

    @classmethod
    def from_file(cls, filename: str | Path):
        with open(filename) as f:
            return cls(f.read())

    def __init__(self, config_str: str):
        config = yaml.safe_load(config_str)

        self.n_vals = sorted(config["grid_points"]["n"])
        self.d_vals = sorted(config["grid_points"]["d"])
        self.h_vals = sorted(config["grid_points"]["h"])

        self.params = [
            [[None for _ in range(len(self.h_vals))] for _ in range(len(self.d_vals))]
            for _ in range(len(self.n_vals))
        ]

        for key, params in config["settings"].items():
            n, h, d = map(float, key.split(","))
            n_idx = self.n_vals.index(n)
            d_idx = self.d_vals.index(d)
            h_idx = self.h_vals.index(h)
            self.params[n_idx][d_idx][h_idx] = params

    def get_parameters(self, n: float, h: float, d: float) -> tuple[int, int, int]:
        # e.g. get the largest index into n_vals st the value is <= n
        n_idx = min(bisect_right(self.n_vals, n), len(self.n_vals)) - 1
        d_idx = min(bisect_right(self.d_vals, d), len(self.d_vals)) - 1
        h_idx = min(bisect_right(self.h_vals, h), len(self.h_vals)) - 1

        return self.params[n_idx][d_idx][h_idx]


class ParamLookup:
    @classmethod
    def from_folder(cls, config_dir: Path, config_name: str):
        device_cap = torch.cuda.get_device_capability()
        caps = [
            tuple(map(int, x.name.split("_")))
            for x in config_dir.iterdir()
            if x.is_dir()
        ]

        if device_cap in caps:
            return cls(config_dir / f"{device_cap[0]}_{device_cap[1]}", config_name)

        major_matches = [cap for cap in caps if cap[0] == device_cap[0]]
        if major_matches:
            compatible_minors = [
                cap[1] for cap in major_matches if cap[1] <= device_cap[1]
            ]
            if compatible_minors:
                return cls(
                    config_dir / f"{device_cap[0]}_{max(compatible_minors)}",
                    config_name,
                )

        compatible_majors = [cap[0] for cap in caps if cap[0] < device_cap[0]]
        if not compatible_majors:
            raise ValueError(
                f"No compatible configuration found for compute capability {device_cap}"
            )

        best_major = max(compatible_majors)
        best_cap = max(cap for cap in caps if cap[0] == best_major)
        return cls(config_dir / f"{best_cap[0]}_{best_cap[1]}", config_name)

    def __init__(self, root: Path, config_name):
        self.root = root
        dtypes = {torch.bfloat16, torch.float32, torch.float16}
        self.lookup = {
            dtype: DtypeParamLookup.from_file(root / f"{dtype}" / config_name)
            for dtype in dtypes
        }

    def __getitem__(self, dtype: torch.dtype) -> DtypeParamLookup:
        return self.lookup[dtype]
