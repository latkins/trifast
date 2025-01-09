import triton
from functools import partial


def get_neighbors(current, allowed_list):
    """Get the next lower and higher values from allowed list"""
    idx = allowed_list.index(current)
    prev_val = allowed_list[idx - 1] if idx > 0 else current
    next_val = allowed_list[idx + 1] if idx < len(allowed_list) - 1 else current
    return [prev_val, current, next_val]


def gen_cfgs(configs, named_args, *, base_configs, **kwargs):
    q = named_args["q_ptr"]

    dtype = q.dtype
    heads, n, _, dim = q.shape

    # Allowed values
    allowed = {
        "block_j": [16, 32, 64, 128, 256],
        "block_k": [16, 32, 64, 128, 256],
        "warps": [1, 2, 4, 8, 16],
        "stages": [1, 2, 3, 4, 5, 6],
    }

    # Find the closest matching configuration
    def find_closest_config():
        # First try exact match
        exact_match = base_configs.get((heads, dim, n, dtype))
        if exact_match:
            return exact_match

        # For bfloat16 and float32, try to match based on n with same heads/dim
        if dtype in ["torch.bfloat16", "torch.float32"]:
            configs = [
                (k, v)
                for k, v in base_configs.items()
                if k[0] == heads and k[1] == dim and k[3] == dtype
            ]
            if configs:
                # Find closest n
                closest = min(configs, key=lambda x: abs(x[0][2] - n))
                return closest[1]

        # If still no match, find most similar configuration
        if dtype == "torch.bfloat16":
            if heads == 1:
                return base_configs[(1, 32, 16, "torch.bfloat16")]
            return base_configs[(4, 32, 128, "torch.bfloat16")]
        elif dtype == "torch.float32":
            return base_configs[(4, 32, 128, "torch.float32")]
        else:
            return base_configs[(4, 32, 32, "torch.float16")]

    # Get base configuration
    base_params = find_closest_config()

    # Generate variants
    variants = []
    for block_j in get_neighbors(base_params["block_j"], allowed["block_j"]):
        for block_k in get_neighbors(base_params["block_k"], allowed["block_k"]):
            for warps in get_neighbors(base_params["warps"], allowed["warps"]):
                for stages in get_neighbors(base_params["stages"], allowed["stages"]):
                    # We should be able to skip values here where blocks are > n, if there is a more
                    # appropriate value in the allowed values.
                    # e.g. if N is 16, we shouldn't allow block_size > 16.
                    variant = triton.Config(
                        kwargs={
                            "BLOCK_J": block_j,
                            "BLOCK_K": block_k,
                        },
                        num_warps=warps,
                        num_stages=stages,
                    )
                    if variant not in variants:
                        variants.append(variant)

    return variants


fwd_base_cfgs = {
    (4, 32, 32, "torch.bfloat16"): {
        "block_j": 32,
        "block_k": 32,
        "warps": 2,
        "stages": 1,
    },
    (4, 32, 64, "torch.bfloat16"): {
        "block_j": 32,
        "block_k": 32,
        "warps": 2,
        "stages": 1,
    },
    (4, 32, 128, "torch.bfloat16"): {
        "block_j": 32,
        "block_k": 32,
        "warps": 2,
        "stages": 2,
    },
    (4, 32, 256, "torch.bfloat16"): {
        "block_j": 64,
        "block_k": 32,
        "warps": 4,
        "stages": 2,
    },
    (4, 32, 512, "torch.bfloat16"): {
        "block_j": 128,
        "block_k": 32,
        "warps": 4,
        "stages": 2,
    },
    (4, 32, 1024, "torch.bfloat16"): {
        "block_j": 128,
        "block_k": 32,
        "warps": 4,
        "stages": 2,
    },
    (4, 32, 2048, "torch.bfloat16"): {
        "block_j": 128,
        "block_k": 32,
        "warps": 4,
        "stages": 1,
    },
    (4, 32, 32, "torch.float32"): {
        "block_j": 16,
        "block_k": 32,
        "warps": 4,
        "stages": 1,
    },
    (4, 32, 64, "torch.float32"): {
        "block_j": 16,
        "block_k": 16,
        "warps": 2,
        "stages": 2,
    },
    (4, 32, 128, "torch.float32"): {
        "block_j": 32,
        "block_k": 16,
        "warps": 2,
        "stages": 2,
    },
    (4, 32, 256, "torch.float32"): {
        "block_j": 64,
        "block_k": 32,
        "warps": 4,
        "stages": 3,
    },
    (4, 32, 512, "torch.float32"): {
        "block_j": 64,
        "block_k": 16,
        "warps": 4,
        "stages": 1,
    },
    (4, 32, 1024, "torch.float32"): {
        "block_j": 128,
        "block_k": 16,
        "warps": 4,
        "stages": 1,
    },
    (4, 32, 2048, "torch.float32"): {
        "block_j": 64,
        "block_k": 32,
        "warps": 4,
        "stages": 3,
    },
    (4, 32, 32, "torch.float16"): {
        "block_j": 16,
        "block_k": 32,
        "warps": 1,
        "stages": 1,
    },
    (1, 16, 16, "torch.bfloat16"): {
        "block_j": 32,
        "block_k": 16,
        "warps": 2,
        "stages": 5,
    },
    (1, 32, 16, "torch.bfloat16"): {
        "block_j": 32,
        "block_k": 16,
        "warps": 2,
        "stages": 2,
    },
    (1, 64, 16, "torch.bfloat16"): {
        "block_j": 16,
        "block_k": 16,
        "warps": 1,
        "stages": 3,
    },
}


gen_fwd_cfgs = partial(gen_cfgs, base_configs=fwd_base_cfgs)

# qkv can mostly use the same cfgs due to similar compute pattern
bwd_kv_cfgs = {
    (4, 32, 32, "torch.bfloat16"): {
        "block_j": 32,
        "block_k": 32,
        "warps": 2,
        "stages": 1,
    },
    (4, 32, 64, "torch.bfloat16"): {
        "block_j": 32,
        "block_k": 32,
        "warps": 2,
        "stages": 1,
    },
    (4, 32, 128, "torch.bfloat16"): {
        "block_j": 32,
        "block_k": 32,
        "warps": 2,
        "stages": 2,
    },
    (4, 32, 256, "torch.bfloat16"): {
        "block_j": 32,
        "block_k": 64,
        "warps": 4,
        "stages": 2,
    },
    (4, 32, 512, "torch.bfloat16"): {
        "block_j": 32,
        "block_k": 128,
        "warps": 4,
        "stages": 2,
    },
    (4, 32, 1024, "torch.bfloat16"): {
        "block_j": 32,
        "block_k": 128,
        "warps": 4,
        "stages": 2,
    },
    (4, 32, 2048, "torch.bfloat16"): {
        "block_j": 32,
        "block_k": 128,
        "warps": 4,
        "stages": 1,
    },
    (4, 32, 32, "torch.float32"): {
        "block_j": 32,
        "block_k": 16,
        "warps": 4,
        "stages": 1,
    },
    (4, 32, 64, "torch.float32"): {
        "block_j": 16,
        "block_k": 16,
        "warps": 2,
        "stages": 2,
    },
    (4, 32, 128, "torch.float32"): {
        "block_j": 16,
        "block_k": 32,
        "warps": 2,
        "stages": 2,
    },
    (4, 32, 256, "torch.float32"): {
        "block_j": 32,
        "block_k": 64,
        "warps": 4,
        "stages": 3,
    },
    (4, 32, 512, "torch.float32"): {
        "block_j": 16,
        "block_k": 64,
        "warps": 4,
        "stages": 1,
    },
    (4, 32, 1024, "torch.float32"): {
        "block_j": 16,
        "block_k": 128,
        "warps": 4,
        "stages": 1,
    },
    (4, 32, 2048, "torch.float32"): {
        "block_j": 32,
        "block_k": 64,
        "warps": 4,
        "stages": 3,
    },
    (4, 32, 32, "torch.float16"): {
        "block_j": 32,
        "block_k": 16,
        "warps": 1,
        "stages": 1,
    },
    (1, 16, 16, "torch.bfloat16"): {
        "block_j": 16,
        "block_k": 32,
        "warps": 2,
        "stages": 5,
    },
    (1, 32, 16, "torch.bfloat16"): {
        "block_j": 16,
        "block_k": 32,
        "warps": 2,
        "stages": 2,
    },
    (1, 64, 16, "torch.bfloat16"): {
        "block_j": 16,
        "block_k": 16,
        "warps": 1,
        "stages": 3,
    },
}
gen_bwd_kv_cfgs = partial(gen_cfgs, base_configs=bwd_kv_cfgs)

# q also loops of K, copy fwd for now.
bwd_q_cfgs = fwd_base_cfgs.copy()
gen_bwd_q_cfgs = partial(gen_cfgs, base_configs=bwd_q_cfgs)

bwd_b_cfgs = fwd_base_cfgs.copy()
gen_bwd_b_cfgs = partial(gen_cfgs, base_configs=bwd_b_cfgs)


gen_bwd_pre_cfgs = lambda *args, **kwargs: [
    triton.Config(
        {"BLOCK_J": block_j},
        num_warps=warps,
        num_stages=stages,
    )
    for block_j in [16, 32, 64]
    for warps in [1, 2, 4]
    for stages in [1, 2, 3]
]
