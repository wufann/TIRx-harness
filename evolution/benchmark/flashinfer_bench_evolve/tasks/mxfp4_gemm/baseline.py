#!/usr/bin/env python3
"""Reference and aiter baseline for the MXFP4 GEMM task (ROCm, gfx950).

Format (OCP MX): A [M, K/2] and B [N, K/2] uint8 hold two e2m1 values per byte, the even
K element in the low nibble; SA [M, K/32] and SB [N, K/32] hold one E8M0 scale
(2^(byte - 127)) per 32 consecutive K elements. D[M, N] = dequant(A) @ dequant(B)^T in bf16.
"""

from __future__ import annotations

import torch

# e2m1 code -> value; bit 3 is the sign.
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def dequantize(packed, scales):
    """[rows, K/2] uint8 codes + [rows, K/32] E8M0 -> [rows, K] float32."""

    table = torch.tensor(_E2M1, dtype=torch.float32, device=packed.device)
    codes = torch.stack((packed & 0xF, packed >> 4), dim=-1).flatten(-2).long()
    values = table[codes]
    scale = torch.exp2(scales.float() - 127.0).repeat_interleave(32, dim=-1)
    return values * scale


@torch.no_grad()
def reference(a, sa, b, sb):
    """Dequantize and multiply in float32; the correctness oracle."""

    return (dequantize(a, sa) @ dequantize(b, sb).T).to(torch.bfloat16)


@torch.no_grad()
def prepare(a, sa, b, sb):
    """aiter's layouts (outside timing): 16x16-shuffled weight, shuffled E8M0 scales."""

    from aiter import dtypes
    from aiter.ops.shuffle import shuffle_weight
    from aiter.utility.fp4_utils import e8m0_shuffle

    return (
        a.view(dtypes.fp4x2),
        shuffle_weight(b.view(dtypes.fp4x2), layout=(16, 16)),
        e8m0_shuffle(sa).view(dtypes.fp8_e8m0),
        e8m0_shuffle(sb).view(dtypes.fp8_e8m0),
    )


@torch.no_grad()
def run_prepared(a, b, sa, sb):
    """aiter's tuned MXFP4 GEMM (asm / CK / triton picked per shape)."""

    from aiter.ops.gemm_op_a4w4 import gemm_a4w4

    return gemm_a4w4(a, b, sa, sb, dtype=torch.bfloat16)


@torch.no_grad()
def run(a, sa, b, sb):
    """Run the complete public task contract through aiter."""

    return run_prepared(*prepare(a, sa, b, sb))
