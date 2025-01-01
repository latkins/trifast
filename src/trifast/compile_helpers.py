import yaml
from functools import lru_cache
from bisect import bisect_right
from pathlib import Path
import torch


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

    @lru_cache(maxsize=128)
    def get_parameters(self, n: float, h: float, d: float) -> tuple[int, int, int]:
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
