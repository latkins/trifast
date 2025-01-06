import functools
import torch
from triton.runtime import driver
from triton.testing import (
    get_max_simd_tflops,
    get_max_tensorcore_tflops,
    nvsmi,
)


@functools.lru_cache()
def get_clock_rate_in_khz():
    # Make this work
    return nvsmi(["clocks.max.sm"])[0] * 1e3


def get_tensorcore_tflops(device, num_ctas, num_warps, dtype):
    """return compute throughput in TOPS"""
    total_warps = num_ctas * min(num_warps, 4)
    num_subcores = (
        driver.active.utils.get_device_properties(device)["multiprocessor_count"] * 4
    )  # on recent GPUs
    tflops = (
        min(num_subcores, total_warps)
        / num_subcores
        * get_max_tensorcore_tflops(dtype, get_clock_rate_in_khz(), device)
    )
    return tflops


def get_simd_tflops(device, num_ctas, num_warps, dtype):
    """return compute throughput in TOPS"""
    total_warps = num_ctas * min(num_warps, 4)
    num_subcores = (
        driver.active.utils.get_device_properties(device)["multiprocessor_count"] * 4
    )  # on recent GPUs
    tflops = (
        min(num_subcores, total_warps)
        / num_subcores
        * get_max_simd_tflops(dtype, get_clock_rate_in_khz(), device)
    )
    return tflops


def get_tflops(device, num_ctas, num_warps, dtype):
    capability = torch.cuda.get_device_capability(device)
    if capability[0] < 8 and dtype == torch.float32:
        return get_simd_tflops(device, num_ctas, num_warps, dtype)
    return get_tensorcore_tflops(device, num_ctas, num_warps, dtype)
