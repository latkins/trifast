import torch
import triton
from trifast.autotune_helpers import get_tflops
from triton.testing import get_dram_gbps


def estimate_fwd(q_ptr, BLOCK_J, BLOCK_K, num_warps, **kwargs):
    """Estimate forward pass runtime in milliseconds."""
    # Get input dimensions and device info
    h, _, n, dim = q_ptr.shape
    device_index = q_ptr.device.index
    dtype = q_ptr.dtype

    # Calculate total number of thread blocks (CTAs)
    grid_m = triton.cdiv(n, BLOCK_J)
    num_ctas = grid_m * n * h

    # Calculate FLOPs
    per_step_flops = 4.0 * BLOCK_J * BLOCK_K * dim
    num_steps = n / BLOCK_K
    total_flops = per_step_flops * num_steps * num_ctas

    # Compute time estimate
    device_tflops = get_tflops(device_index, num_ctas, num_warps, dtype)
    compute_time_ms = (total_flops / (device_tflops * 1e12)) * 1e3

    # Memory time estimate
    dram_gbps = get_dram_gbps()
    bytes_per_el = {torch.float16: 2, torch.bfloat16: 2, torch.float32: 4}.get(dtype, 4)
    total_bytes = (
        8.0 * h * n * dim * bytes_per_el
    )  # Factor of 8 for Q,K,V,O reads/writes
    mem_time_ms = (total_bytes / (dram_gbps * (1 << 30))) * 1e3

    return max(compute_time_ms, mem_time_ms)


def estimate_bwd_preprocess(o_ptr, BLOCK_J, num_warps, overhead_constant=1.0, **kwargs):
    """Estimate preprocessing step runtime in milliseconds."""
    # Get input dimensions and device info
    h, i, n, dim = o_ptr.shape
    device_index = o_ptr.device.index
    dtype = o_ptr.dtype

    # Calculate number of thread blocks
    num_ctas = triton.cdiv(n, BLOCK_J) * i * h

    # Calculate FLOPs (multiply and reduction operations)
    per_cta_flops = 2.0 * BLOCK_J * dim * overhead_constant
    total_flops = per_cta_flops * num_ctas

    # Compute time estimate
    device_tflops = get_tflops(device_index, num_ctas, num_warps, dtype)
    compute_time_ms = (total_flops / (device_tflops * 1e12)) * 1e3

    # Memory time estimate
    dram_gbps = get_dram_gbps()
    bytes_per_el = {torch.float16: 2, torch.bfloat16: 2, torch.float32: 4}.get(dtype, 4)
    total_bytes = (2.0 * BLOCK_J * dim + BLOCK_J) * bytes_per_el * num_ctas

    mem_time_ms = (total_bytes / (dram_gbps * (1 << 30))) * 1e3
    return max(compute_time_ms, mem_time_ms)


def estimate_bwd_kv(
    q_ptr, BLOCK_J, BLOCK_K, num_warps=4, overhead_constant=2.0, **kwargs
):
    """Estimate K/V gradients computation runtime in milliseconds."""
    # Get input dimensions and device info
    h, i, n, dim = q_ptr.shape
    device_index = q_ptr.device.index
    dtype = q_ptr.dtype

    # Calculate thread blocks and steps
    num_ctas = triton.cdiv(n, BLOCK_K) * i * h
    num_steps = float(n) / float(BLOCK_J)

    # Calculate FLOPs (3 matrix multiplications per step)
    per_step_flops = 6.0 * BLOCK_J * BLOCK_K * dim * overhead_constant
    total_flops = per_step_flops * num_steps * num_ctas

    # Compute time estimate
    device_tflops = get_tflops(device_index, num_ctas, num_warps, dtype)
    compute_time_ms = (total_flops / (device_tflops * 1e12)) * 1e3

    # Memory time estimate
    dram_gbps = get_dram_gbps()
    bytes_per_el = {torch.float16: 2, torch.bfloat16: 2, torch.float32: 4}.get(dtype, 4)
    elements_per_step = 4.0 * BLOCK_J * BLOCK_K * dim * overhead_constant
    total_bytes = elements_per_step * num_steps * num_ctas * bytes_per_el

    mem_time_ms = (total_bytes / (dram_gbps * (1 << 30))) * 1e3
    return max(compute_time_ms, mem_time_ms)


def estimate_bwd_q(q_ptr, BLOCK_J, BLOCK_K, num_warps, overhead_constant=2.0, **kwargs):
    """Estimate Q gradients computation runtime in milliseconds."""
    # Get input dimensions and device info
    h, i, n, dim = q_ptr.shape
    device_index = q_ptr.device.index
    dtype = q_ptr.dtype

    # Calculate thread blocks and steps
    num_ctas = triton.cdiv(n, BLOCK_J) * i * h
    num_steps = float(n) / float(BLOCK_K)

    # Calculate FLOPs (3 matrix multiplications per step)
    per_step_flops = 6.0 * BLOCK_J * BLOCK_K * dim * overhead_constant
    total_flops = per_step_flops * num_steps * num_ctas

    # Compute time estimate
    device_tflops = get_tflops(device_index, num_ctas, num_warps, dtype)
    compute_time_ms = (total_flops / (device_tflops * 1e12)) * 1e3

    # Memory time estimate
    dram_gbps = get_dram_gbps()
    bytes_per_el = {torch.float16: 2, torch.bfloat16: 2, torch.float32: 4}.get(dtype, 4)
    elements_per_step = 6.0 * BLOCK_J * BLOCK_K * dim * overhead_constant
    total_bytes = elements_per_step * num_steps * num_ctas * bytes_per_el

    mem_time_ms = (total_bytes / (dram_gbps * (1 << 30))) * 1e3
    return max(compute_time_ms, mem_time_ms)


def estimate_bwd_b(q_ptr, BLOCK_J, BLOCK_K, num_warps, overhead_constant=2.0, **kwargs):
    """Estimate bias gradients computation runtime in milliseconds."""
    # Get input dimensions and device info
    h, n, _, dim = q_ptr.shape
    device_index = q_ptr.device.index
    dtype = q_ptr.dtype

    # Calculate thread blocks and iterations
    grid_j = triton.cdiv(n, BLOCK_J)
    grid_k = triton.cdiv(n, BLOCK_K)
    num_ctas = grid_j * grid_k * h

    # Calculate FLOPs (2 matrix multiplications per iteration)
    per_iter_flops = 6.0 * BLOCK_J * BLOCK_K * dim * overhead_constant
    total_flops = per_iter_flops * float(n) * num_ctas

    # Compute time estimate
    device_tflops = get_tflops(device_index, num_ctas, num_warps, dtype)
    compute_time_ms = (total_flops / (device_tflops * 1e12)) * 1e3

    # Memory time estimate
    dram_gbps = get_dram_gbps()
    bytes_per_el = {torch.float16: 2, torch.bfloat16: 2, torch.float32: 4}.get(dtype, 4)
    elements_per_iter = 6.0 * BLOCK_J * BLOCK_K * dim * overhead_constant
    total_bytes = elements_per_iter * float(n) * num_ctas * bytes_per_el

    mem_time_ms = (total_bytes / (dram_gbps * (1 << 30))) * 1e3
    return max(compute_time_ms, mem_time_ms)
