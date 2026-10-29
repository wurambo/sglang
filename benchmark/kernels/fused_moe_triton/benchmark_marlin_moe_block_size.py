"""Sweep Marlin MoE M-block sizes for a configurable (experts, top-k, tokens) grid.

Times fused_marlin_moe end to end (moe_align + both grouped GEMMs + activation
+ reduce) while forcing each candidate block size, and marks what the selector
picks, so block-size heuristic changes can be validated on any GPU.

EP is modeled the way the standard dispatcher presents it to the runner: top-k
ids are drawn over the global expert space, assignments landing on this rank
keep a local id in [0, num_local_experts) and the rest become the -1 drop
sentinel.

Example (256 global experts, top-8, EP8 -> 32 local experts):
    python benchmark_marlin_moe_block_size.py \
        --num-experts 256 --top-k 8 --num-local-experts 32 \
        --hidden-size 4096 --intermediate-size 2048 \
        --tokens 1 8 16 64 128 256 512 1024 4096
"""

import argparse
import statistics

import torch
from sgl_kernel.scalar_type import scalar_types

import sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe as fused_marlin_moe_module
from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
    _MOE_BLOCK_SIZES,
    fused_marlin_moe,
    select_marlin_moe_block_size,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.test_marlin_utils import marlin_quantize

set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))


def quantize_experts(w: torch.Tensor, group_size: int):
    qweights, scales = [], []
    for i in range(w.shape[0]):
        _, qweight, scale, _, _, _ = marlin_quantize(
            w[i].transpose(1, 0), scalar_types.uint4b8, group_size, act_order=False
        )
        qweights.append(qweight)
        scales.append(scale)
    return torch.stack(qweights).contiguous(), torch.stack(scales)


def make_topk(num_tokens, num_experts, num_local_experts, top_k):
    score = torch.randn((num_tokens, num_experts), dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(torch.softmax(score, dim=-1), top_k, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    # Localize like the standard dispatcher: this rank owns global experts
    # [0, num_local_experts); remote assignments become the -1 drop sentinel.
    topk_ids = torch.where(topk_ids < num_local_experts, topk_ids, -1)
    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


def force_block(block):
    def _select(num_tokens, top_k, global_num_experts):
        return block

    return _select


def time_call(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1e3)  # us
    return statistics.median(times)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-experts", type=int, default=256, help="global experts")
    parser.add_argument(
        "--num-local-experts",
        type=int,
        default=None,
        help="experts on this rank (default: --num-experts, i.e. no EP)",
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-size", type=int, default=2048)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[1, 8, 16, 64, 128, 256, 512, 1024]
    )
    parser.add_argument(
        "--block-sizes", type=int, nargs="+", default=list(_MOE_BLOCK_SIZES)
    )
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.set_default_device("cuda")
    dtype = getattr(torch, args.dtype)
    e_local = args.num_local_experts or args.num_experts
    k, n = args.hidden_size, args.intermediate_size

    w1 = torch.randn((e_local, 2 * n, k), dtype=dtype) / 20
    w2 = torch.randn((e_local, k, n), dtype=dtype) / 20
    qweight1, scales1 = quantize_experts(w1, args.group_size)
    qweight2, scales2 = quantize_experts(w2, args.group_size)

    print(f"device={torch.cuda.get_device_name()}")
    print(
        f"experts={args.num_experts} local={e_local} top_k={args.top_k} "
        f"hidden={k} intermediate={n} dtype={args.dtype}"
    )
    header = ["tokens"] + [f"b={b}" for b in args.block_sizes] + ["selected"]
    print("  ".join(f"{h:>10}" for h in header))

    for num_tokens in args.tokens:
        a = torch.randn((num_tokens, k), dtype=dtype) / 10
        score = torch.randn((num_tokens, e_local), dtype=dtype)
        topk_weights, topk_ids = make_topk(
            num_tokens, args.num_experts, e_local, args.top_k
        )

        def run():
            return fused_marlin_moe(
                a,
                qweight1,
                qweight2,
                scales1,
                scales2,
                score,
                topk_weights,
                topk_ids,
                global_num_experts=args.num_experts,
                num_bits=4,
            )

        selected = select_marlin_moe_block_size(
            num_tokens=num_tokens,
            top_k=args.top_k,
            global_num_experts=args.num_experts,
        )
        row = [f"{num_tokens:>10}"]
        baseline = None
        for block in args.block_sizes:
            fused_marlin_moe_module.select_marlin_moe_block_size = force_block(block)
            try:
                out = run()
                if baseline is None:
                    baseline = out.float()
                else:
                    rel = (
                        (out.float() - baseline).norm()
                        / baseline.norm().clamp_min(1e-6)
                    ).item()
                    assert rel < 1e-2, f"block={block} rel diff {rel:.3e}"
                us = time_call(run, args.warmup, args.iters)
                row.append(f"{us:>10.1f}")
            finally:
                fused_marlin_moe_module.select_marlin_moe_block_size = (
                    select_marlin_moe_block_size
                )
        row.append(f"{selected:>10}")
        print("  ".join(row))


if __name__ == "__main__":
    main()
