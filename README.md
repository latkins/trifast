Fused Triangle Self Attention kernel, written in triton. Basically flash attention, but for triangle self attention.
Implementation heavily inspired by [FlagAttention](https://github.com/FlagOpen/FlagAttention/tree/main) and the [triton fused attention tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html#sphx-glr-getting-started-tutorials-06-fused-attention-py).

- n^2 memory complexity (vs n^3 for pure pytorch).
- Faster (~2x) backward pass than next fastest implementation I could find (DS4S evoformer kernel).
- Faster (~4x) forward pass than next fastest implementation I could find (DS4S evoformer kernel).
- As far as I can tell, faster than naieve implementation.

## Plots
All done on a 3090 in bfloat16.
### Forward
![TSA forward runtime](benchmark/NVIDIA-GeForce-RTX-3090/runtime/tri_attn_fwd_torch.bfloat16.png "TSA forward runtime")
![TSA forward memory](benchmark/NVIDIA-GeForce-RTX-3090/memory/peak_memory_fwd_torch.bfloat16.png "TSA forward memory")

Backward
![TSA backward runtime](benchmark/NVIDIA-GeForce-RTX-3090/runtime/tri_attn_bwd_torch.bfloat16.png "TSA backward runtime")
![TSA backward memory](benchmark/NVIDIA-GeForce-RTX-3090/memory/peak_memory_bwd_torch.bfloat16.png "TSA backward memory")


Todos:
- [] Try to train a model with it.
- [] Can we perform and of dq/db/dkv transposed?
- [] Rewrite autotuner
