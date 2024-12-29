import yaml
from functools import lru_cache
from bisect import bisect_right
from pathlib import Path


class ParamLookup:
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
