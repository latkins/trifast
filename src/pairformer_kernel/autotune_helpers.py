from enum import Enum
from typing import Union


class DataType(Enum):
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    INT8 = "int8"


def get_dtype_size(dtype: Union[DataType, str]) -> int:
    """
    Get size in bytes for different datatypes.

    Args:
        dtype: DataType enum or string representing the datatype

    Returns:
        Size in bytes for the datatype
    """
    if isinstance(dtype, str):
        dtype = DataType(dtype)

    dtype_sizes = {
        DataType.FLOAT32: 4,
        DataType.FLOAT16: 2,
        DataType.BFLOAT16: 2,
        DataType.INT8: 1,
    }
    return dtype_sizes[dtype]


def shared_mem_per_block(config, N: int, DIM: int, dtype: Union[DataType, str]):
    """
    Calculate shared memory usage per block accounting for datatype.

    Args:
        config: Configuration object containing BLOCK_J, BLOCK_K, and num_stages
        N: Sequence length
        DIM: Hidden dimension size
        dtype: DataType for the tensors

    Returns:
        Total shared memory usage in bytes
    """
    block_j = config["BLOCK_J"]
    block_k = config["BLOCK_K"]
    num_stages = config.num_stages
    dtype_size = get_dtype_size(dtype)

    # Calculate memory needed for each buffer
    # Note: scores buffer always uses float32 for numerical stability
    q_buffer = block_j * DIM * dtype_size
    k_buffer = block_k * DIM * dtype_size
    v_buffer = block_k * DIM * dtype_size
    scores_buffer = block_j * block_k * 4  # Always float32 for attention scores

    # Account for pipeline buffering
    pipeline_buffers = (k_buffer + v_buffer) * num_stages

    total_shared = q_buffer + pipeline_buffers + scores_buffer

    return total_shared


def estimate_performance(config, N: int, DIM: int, dtype: str = "bfloat16"):
    """
    Estimate performance considering datatype characteristics.

    Args:
        config: Configuration object with BLOCK_J, BLOCK_K, and num_warps
        N: Sequence length
        DIM: Hidden dimension size
        dtype: DataType for the tensors

    Returns:
        Estimated TFLOPS or equivalent compute throughput
    """
    block_j = config["BLOCK_J"]
    block_k = config["BLOCK_K"]
    dtype_size = get_dtype_size(dtype)

    # Calculate shared memory requirement
    shared_mem = shared_mem_per_block(config, N, DIM, dtype)

    # Adjust max shared memory based on GPU architecture
    # This could be made more sophisticated based on detected GPU
    max_shared_mem_per_sm = 48000  # 48KB default
    blocks_per_sm_shared = max_shared_mem_per_sm // shared_mem

    # Adjust register pressure based on dtype
    # Smaller datatypes typically need fewer registers
    reg_multiplier = {
        DataType.FLOAT32: 1.0,
        DataType.FLOAT16: 0.7,  # Approximate reduction in register pressure
        DataType.BFLOAT16: 0.7,
        DataType.INT8: 0.5,
    }[DataType(dtype)]

    reg_estimate = int(block_j * DIM * 4 * reg_multiplier)

    # Calculate theoretical occupancy with dtype considerations
    warps_per_sm = min(
        48 // config.num_warps,  # Warp slots per SM
        65536 // reg_estimate,  # Adjusted register limitation
        blocks_per_sm_shared * config.num_warps,  # Shared memory limitation
    )

    # Adjust compute throughput based on dtype
    # Some GPUs have different throughput for different precision
    compute_multiplier = {
        DataType.FLOAT32: 1.0,
        DataType.FLOAT16: 2.0,  # Often 2x throughput vs FP32
        DataType.BFLOAT16: 2.0,
        DataType.INT8: 4.0,  # Often 4x throughput vs FP32
    }[DataType(dtype)]

    # Base compute throughput calculation
    ops_per_warp = (block_j * block_k * DIM * 2) / (block_j * block_k / 32)

    # Memory bandwidth calculation adjusted for dtype
    mem_bytes_per_block = (block_j + block_k) * DIM * dtype_size

    # Adjust bandwidth requirements based on dtype
    # Smaller datatypes reduce memory pressure
    bandwidth_required = mem_bytes_per_block * warps_per_sm
    mem_bandwidth_factor = min(1.0, 900 / bandwidth_required)  # Assuming ~900GB/s peak

    theoretical_tflops = (
        warps_per_sm * ops_per_warp * mem_bandwidth_factor * compute_multiplier
    )

    return theoretical_tflops


def config_pruner(configs, named_args, **kwargs):
    """Remove inappropriate configs based on input shapes."""
    N = kwargs["N"]
    DIM = kwargs["DIM"]

    pruned_configs = []
    for config in configs:
        BLOCK_J = config.kwargs["BLOCK_J"]
        BLOCK_K = config.kwargs["BLOCK_K"]
        num_warps = config.num_warps

        accept = True

        # for small N, remove big blocks
        accept &= BLOCK_J <= N
        accept &= BLOCK_K <= N

        # for large N, remove small blocks
        if N > 128:
            accept &= BLOCK_J > 16


        if N > 256:
            accept &= num_warps > 2


        if accept:
            pruned_configs.append(config)

    return pruned_configs
