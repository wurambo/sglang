"""Contract tests for the Marlin MoE M-block-size heuristic.

The selector runs host-side at schedule time, so it may depend only on scalars
(never a tensor), and it must key on expected tokens per expert over the global
expert space: under EP the local expert count overestimates occupancy by the EP
factor and selects tiles that are mostly padding.
"""

import sys

import pytest

from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
    select_marlin_moe_block_size,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


@pytest.mark.parametrize("top_k", [2, 8, 16])
@pytest.mark.parametrize("global_num_experts", [64, 256, 896])
def test_monotonic_in_token_count(top_k, global_num_experts):
    prev = None
    for num_tokens in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384]:
        block = select_marlin_moe_block_size(
            num_tokens=num_tokens, top_k=top_k, global_num_experts=global_num_experts
        )
        if prev is not None:
            assert block >= prev
        prev = block


@pytest.mark.parametrize(
    ("num_tokens", "top_k", "num_experts", "expected"),
    [
        (1, 1, 8, 8),
        (16, 2, 8, 8),
        (64, 2, 8, 32),
        (256, 2, 8, 64),
        (64, 8, 64, 16),
        (256, 8, 64, 48),
        (1024, 8, 64, 64),
        (8192, 1, 16, 64),
    ],
)
def test_no_ep_selection_is_unchanged(num_tokens, top_k, num_experts, expected):
    """Without EP the global and local counts coincide, so these are the blocks
    the pre-existing occupancy loop selected."""
    assert (
        select_marlin_moe_block_size(
            num_tokens=num_tokens, top_k=top_k, global_num_experts=num_experts
        )
        == expected
    )


def test_ep_selects_by_global_expert_count():
    """256 global experts, top-8, EP8 (32 local experts): at 128-256 tokens the
    per-expert occupancy is 4-8 rows, so tiles of 48-64 rows (what dividing by
    the local count selected) are mostly padding."""
    assert (
        select_marlin_moe_block_size(num_tokens=128, top_k=8, global_num_experts=256)
        == 8
    )
    assert (
        select_marlin_moe_block_size(num_tokens=256, top_k=8, global_num_experts=256)
        == 16
    )


@pytest.mark.parametrize(
    ("num_tokens", "expected"),
    [(64, 8), (512, 16), (1024, 32), (2048, 48), (4096, 64), (16384, 64)],
)
def test_high_expert_count_tracks_tokens_per_expert(num_tokens, expected):
    """Kimi-K3-like shape (896 global experts, top-16): the block grows with
    tokens-per-expert (num_tokens / 56 here), not with the raw token count."""
    assert (
        select_marlin_moe_block_size(
            num_tokens=num_tokens, top_k=16, global_num_experts=896
        )
        == expected
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
