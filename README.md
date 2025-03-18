# TriFast

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Triton-based implementation of Fused Triangle Self Attention kernels. TriFast provides an optimized version triangle self attention, using the ideas of flash attention.

## 🚀 Features

- **Memory Efficient**: Achieves n² memory complexity (compared to n³ for pure PyTorch implementations)
- **High Performance**:
  - ⚡ ~4x faster forward pass than the next fastest implementation (DS4S evoformer kernel)
  - ⚡ ~2x faster backward pass than the next fastest implementation (DS4S evoformer kernel)
- **Multiple Precision Support**: Works with float32, bfloat16, and fp16 data types
- **GPU Accelerated**: Benchmarked on NVIDIA GPUs with excellent scaling properties
- **Auto-tuning**: Includes built-in kernel tuning capabilities to optimize for specific workloads

## 📊 Benchmarks

All benchmarks were performed on an NVIDIA GeForce RTX 3090 GPU using BFloat16 precision.

### Forward Pass Performance

#### Runtime
![TSA forward runtime](benchmark/NVIDIA-GeForce-RTX-3090/runtime/tri_attn_fwd_torch.bfloat16.png "TSA forward runtime")

#### Memory Usage
![TSA forward memory](benchmark/NVIDIA-GeForce-RTX-3090/memory/peak_memory_fwd_torch.bfloat16.png "TSA forward memory")

### Backward Pass Performance

#### Runtime
![TSA backward runtime](benchmark/NVIDIA-GeForce-RTX-3090/runtime/tri_attn_bwd_torch.bfloat16.png "TSA backward runtime")

#### Memory Usage
![TSA backward memory](benchmark/NVIDIA-GeForce-RTX-3090/memory/peak_memory_bwd_torch.bfloat16.png "TSA backward memory")

## 🛠️ Installation

```bash
pip install trifast
```

## 📖 Usage

Basic usage of the triangle attention function:

```python
import torch
from trifast import triangle_attention
from trifast.utils import gen_tensors

# Generate tensors (query, key, value, bias, mask)
q, k, v, bias, mask = gen_tensors(n, d, h, use_mask=True, device=device, dtype=dtype, std=scale)

# Apply triangle self attention
out = triangle_attention(q, k, v, bias, mask)
```

## ⚙️ Auto-tuning

TriFast modifies `triton.autotune` to cache the best config to disk. This means the first time the kernel is run will generally be slower than subsequent times (for a given input shape).

### Using the auto-tuner

The package provides a command-line script for auto-tuning:

```bash
# Basic usage
trifast-tune

# For a more extensive tuning process
TRIFAST_FORCE_TUNE=1 trifast-tune

# With custom parameters
trifast-tune --min-n 32 --max-n 2048 --dtype bfloat16 --h 4,8 --d 32,64
```

### Auto-tuner Parameters

```
--min-n          Minimum sequence length to tune (default: 16)
--max-n          Maximum sequence length to tune (default: 1024)
--dtype          PyTorch datatypes to use (comma-separated).
                 Options: float32, bfloat16, float16, all (default: bfloat16)
--h              List of number of heads (comma-separated integers, e.g., "1,2,4") (default: 4)
--d              List of dimensions (comma-separated integers, e.g., "16,32,64") (default: 32)
```

After tuning, the best configurations are cached to disk using `platformdirs` (e.g. under `~/.config/trifast/<version>` on Linux).

## 🧠 How It Works

TriFast implements Triangle Self Attention using Triton to create optimized GPU kernels. It's essentially a Flash Attention applied to triangle self attention, resulting in significant performance gains compared to naive PyTorch implementations.

## 🔍 Implementation Details

This implementation draws inspiration from:
- [FlagAttention](https://github.com/FlagOpen/FlagAttention/tree/main)
- [Triton Fused Attention Tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)
- [Flash Attention](https://arxiv.org/abs/2205.14135)

## 🗺️ Roadmap

- [ ] Explore performance optimizations for dq/db/dkv transposed operations
- [ ] Implement pipelined writes to global memory in backward kernels

## 👨‍💻 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- The FlagOpen team for their work on FlagAttention
- The Triton team for providing excellent documentation and tutorials
- The DS4S team for their evoformer kernel implementation used in benchmarking
