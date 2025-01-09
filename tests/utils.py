import os
import numpy as np
import random
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed=42):
    """
    Fix all random seeds we use for reproducibility.
    """
    # Python random seed
    random.seed(seed)
    # Numpy random seed
    np.random.seed(0)
    # PyTorch random seed
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

        # PyTorch backend settings
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def compare_values(tri, pt, ref, msg="", eps=1e-4):
    a = (tri.float() - ref.float()).abs().max().item()
    b = (pt.float() - ref.float()).abs().max().item() + eps

    # This factor of 3 is pretty arbitrary.
    assert a <= 3 * b, f"{msg} value mismatch, tri: {a:.3e}, pt: {b:.3e}"


def dot(a, b):
    return torch.dot(a.float().flatten(), b.float().flatten())


def cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps=1e-8) -> float:
    """
    Flatten a and b, compute the dot product over
    the product of their magnitudes, and return the scalar.
    """
    a_flat = a.flatten()
    b_flat = b.flatten()
    dot = torch.dot(a_flat, b_flat)
    norm_a = torch.norm(a_flat)
    norm_b = torch.norm(b_flat)
    return (dot / (norm_a * norm_b + eps)).item()


def compare_directions(tensor_kernel, tensor_pt, tensor_ref, msg="", threshold=0.99):
    """
    Compare directions (via dot product / cosine similarity).
    Assert that both kernel and pt are above a threshold similarity to ref.
    """
    cs_kernel_ref = cosine_similarity(tensor_kernel.float(), tensor_ref.float())
    cs_pt_ref = cosine_similarity(tensor_pt.float(), tensor_ref.float())

    assert (
        cs_kernel_ref > threshold
    ), f"{msg} kernel->ref direction mismatch: {cs_kernel_ref:.3f}"
    assert cs_pt_ref > threshold, f"{msg} pt->ref direction mismatch: {cs_pt_ref:.3f}"


def compare_relative_direction(
    tensor_kernel, tensor_pt, tensor_ref, msg="", ratio=0.99
):
    """
    Make sure kernel->ref direction alignment is close to pt->ref direction alignment.
    i.e. cos_sim(kernel, ref) >= ratio * cos_sim(pt, ref).
    """
    cs_kernel_ref = cosine_similarity(tensor_kernel.float(), tensor_ref.float())
    cs_pt_ref = cosine_similarity(tensor_pt.float(), tensor_ref.float())

    assert cs_kernel_ref >= ratio * cs_pt_ref, (
        f"{msg} kernel->ref relative direction mismatch. "
        f"Got cos_sim(kernel, ref)={cs_kernel_ref:.4f} vs. ratio * cos_sim(pt, ref)={ratio*cs_pt_ref:.4f}"
    )
