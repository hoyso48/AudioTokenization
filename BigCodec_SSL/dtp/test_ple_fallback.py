import argparse
import re
import math
import random
from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from tome_ops import PLETopK


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def normalize_last_dim(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return F.normalize(x, dim=-1, eps=eps)


def gen_constant(B: int, N: int, C: int, value: float = 1.0) -> torch.Tensor:
    x = torch.full((B, N, C), float(value))
    return x


def gen_all_equal_vectors(B: int, N: int, C: int) -> torch.Tensor:
    # Same feature vector repeated across time; per-batch different vector
    base = torch.randn(B, 1, C)
    x = base.expand(B, N, C).clone()
    return x


def gen_random_gaussian(B: int, N: int, C: int, scale: float = 1.0) -> torch.Tensor:
    x = torch.randn(B, N, C) * float(scale)
    return x


def gen_sine_with_noise(B: int, N: int, C: int) -> torch.Tensor:
    # Similar to generate_test_signal in the notebook snippet
    t = torch.linspace(0, 6 * math.pi, N)
    batch_tensors: List[torch.Tensor] = []
    for b in range(B):
        base_signal = torch.sin(t + b * 0.5) * 0.8 + torch.cos((t + b * 0.2) * 2.5) * 0.6
        channels: List[torch.Tensor] = []
        for _ in range(C):
            noise = torch.randn(N) * 0.15
            channels.append(base_signal + noise)
        sample = torch.stack(channels, dim=-1)
        batch_tensors.append(sample)
    x = torch.stack(batch_tensors, dim=0)
    return x


def gen_step_segments(B: int, N: int, C: int, num_segments: int = 4) -> torch.Tensor:
    if N == 0:
        return torch.zeros(B, 0, C)
    seg_len = max(1, N // max(1, num_segments))
    vals = torch.randn(B, num_segments, C)
    x_list: List[torch.Tensor] = []
    idx = 0
    for s in range(num_segments):
        run = min(seg_len, N - idx)
        if run <= 0:
            break
        x_list.append(vals[:, s:s+1, :].expand(B, run, C))
        idx += run
    if idx < N:
        pad = vals[:, -1:, :].expand(B, N - idx, C)
        x_list.append(pad)
    x = torch.cat(x_list, dim=1) if len(x_list) > 0 else torch.zeros(B, 0, C)
    return x


def gen_repeated_two_values(B: int, N: int, C: int) -> torch.Tensor:
    a = torch.randn(B, 1, C)
    b = torch.randn(B, 1, C)
    pattern = torch.cat([a, b], dim=1)  # (B, 2, C)
    reps = (N + 1) // 2
    x = pattern.repeat(1, reps, 1)[:, :N, :]
    return x


@dataclass
class Case:
    name: str
    B: int
    N: int
    C: int
    r: float
    generator: Callable[[int, int, int], torch.Tensor]


@dataclass
class Result:
    case: Case
    status: str  # "OK" or "ERROR"
    message: str
    kept_expected: int
    kept_actual: int


def run_case(case: Case, device: str = "cpu") -> Result:
    B, N, C = case.B, case.N, case.C
    x = case.generator(B, N, C).to(device)
    x = normalize_last_dim(x)
    model = PLETopK(case.r).to(device)

    # Algorithm keeps token 0 always: clamp K<=N-1
    kept_expected = int(N - min(math.floor(case.r * N), max(N - 1, 0)))
    try:
        merged_x, btree, avg = model.compute_merge(x)
        kept_actual = int(merged_x.shape[1])

        # Validate btree shape and semantics
        assert btree.shape == (B, N)
        # Validate avg shape and zeros
        assert avg.shape == (B,)
        if avg.abs().sum().item() != 0.0:
            return Result(case, "ERROR", "avg_sim is not all zeros", kept_expected, kept_actual)

        # Validate direct_to_root -> merge/unmerge roundtrip
        root_map = model.btree_to_root_map(btree)
        x_kept = model.merge(x, root_map)
        if not torch.allclose(x_kept, merged_x, atol=1e-5, rtol=1e-5):
            return Result(case, "ERROR", "merge(direct_to_root_map) != merged_x", kept_expected, kept_actual)

        # Unmerge should restore length N
        x_restore = model.unmerge(merged_x, root_map)
        if x_restore.shape[1] != N:
            return Result(case, "ERROR", "unmerge length mismatch", kept_expected, kept_actual)

        if kept_actual != kept_expected:
            # Provide diagnostic details for this mismatch
            diag = analyze_bins(x, case.r)
            return Result(case, "ERROR", f"kept count mismatch without exception | {diag}", kept_expected, kept_actual)

        return Result(case, "OK", "", kept_expected, kept_actual)
    except RuntimeError as e:
        # Enrich with bin analysis
        diag = analyze_bins(x, case.r)
        msg = str(e) + " | " + diag
        return Result(case, "ERROR", msg, kept_expected, kept_actual=-1)
    except Exception as e:
        diag = analyze_bins(x, case.r)
        return Result(case, "ERROR", f"Unexpected: {e} | {diag}", kept_expected, kept_actual=-1)


def analyze_bins(x: torch.Tensor, r: float) -> str:
    """
    Mirror the binning math inside PLETopK to understand failure modes.
    Returns a compact string with per-batch stats (min/max) to keep output readable.
    """
    B, N, C = x.shape
    if N == 0:
        return "N=0"
    K = int(min(max(math.floor(r * N), 0), N - 1))
    M = int(N - K)
    interior = max(M - 1, 0)

    x_norm = F.normalize(x, dim=-1)
    if N > 1:
        d = 1.0 - (x_norm[:, :-1, :] * x_norm[:, 1:, :]).sum(dim=-1)  # (B, N-1)
        w = d + 1e-12
        D = torch.zeros(B, N, device=x.device, dtype=x.dtype)
        D[:, 1:] = torch.cumsum(w, dim=1)
        L = D[:, -1]
    else:
        d = torch.zeros(B, 0, device=x.device, dtype=x.dtype)
        D = torch.zeros(B, 1, device=x.device, dtype=x.dtype)
        L = torch.zeros(B, device=x.device, dtype=x.dtype)

    # Bin width t; use same handling as implementation (avoid div0 with ones)
    t = torch.where(L > 0, L / float(M if M > 0 else 1), torch.ones_like(L))

    # Pair-to-bin assignment
    if N > 1 and M > 0:
        pair_right = D[:, 1:]
        bin_idx = torch.clamp((pair_right / t.view(B, 1)).floor().long(), min=0, max=max(M - 1, 0))
    else:
        bin_idx = torch.zeros(B, max(N - 1, 0), dtype=torch.long, device=x.device)

    # Compute per-bin occupancy excluding pair 0 (pair 0 is globally masked)
    if N > 1 and interior > 0:
        bins = torch.arange(1, M, device=x.device).view(1, -1, 1)  # (1, interior, 1)
        bi = bin_idx.unsqueeze(1)  # (B, 1, N-1)
        mask = (bi == bins)  # (B, interior, N-1)
        # Exclude pair 0
        pair_ids = torch.arange(N - 1, device=x.device).view(1, 1, -1)
        not_pair0 = (pair_ids != 0)
        mask_excl0 = mask & not_pair0
        occ = mask_excl0.sum(dim=2)  # (B, interior)
        empty_bins = (occ == 0)
        num_empty_per_b = empty_bins.sum(dim=1)
        occ_min = int(occ.min().item()) if occ.numel() > 0 else 0
        occ_max = int(occ.max().item()) if occ.numel() > 0 else 0
        empty_min = int(num_empty_per_b.min().item()) if num_empty_per_b.numel() > 0 else 0
        empty_max = int(num_empty_per_b.max().item()) if num_empty_per_b.numel() > 0 else 0
    else:
        occ_min = 0
        occ_max = 0
        empty_min = 0
        empty_max = 0

    L_min = float(L.min().item()) if L.numel() > 0 else 0.0
    L_max = float(L.max().item()) if L.numel() > 0 else 0.0
    t_min = float(t.min().item()) if t.numel() > 0 else 0.0
    t_max = float(t.max().item()) if t.numel() > 0 else 0.0

    return (
        f"B={B} N={N} M={M} interior={interior} | L[min,max]=[{L_min:.3e},{L_max:.3e}] "
        f"t[min,max]=[{t_min:.3e},{t_max:.3e}] | empty_bins_per_batch[min,max]=[{empty_min},{empty_max}] "
        f"occ_per_bin[min,max]=[{occ_min},{occ_max}]"
    )


def build_cases() -> List[Case]:
    Ns = [8, 16, 33, 64, 127]
    Cs = [16, 32]
    Bs = [1, 4]
    # Exclude r < 1/N and r >= 1.0; r candidates will be filtered per N below
    r_candidates = [0.25, 0.33, 0.5, 0.75, 0.9]

    gens: List[Tuple[str, Callable[[int, int, int], torch.Tensor]]] = [
        # ("constant", gen_constant),
        # ("all_equal", gen_all_equal_vectors),
        ("gaussian", gen_random_gaussian),
        ("sine", gen_sine_with_noise),
        # ("steps", gen_step_segments),
        # ("repeat_ab", gen_repeated_two_values),
    ]

    cases: List[Case] = []
    for name, gen in gens:
        for B in Bs:
            for C in Cs:
                for N in Ns:
                    # Build valid r list for this N
                    valid_rs: List[float] = []
                    if N >= 2:
                        r_min = 1.0 / float(N)
                        for r in r_candidates:
                            if (r >= r_min) and (r < 1.0):
                                valid_rs.append(r)
                    elif N == 1:
                        # Only trivial keep-all; skip as r guard disallows r>=1 or r<1/N
                        valid_rs = []
                    else:
                        valid_rs = []

                    for r in valid_rs:
                        cases.append(Case(name=name, B=B, N=N, C=C, r=r, generator=gen))
    return cases


def summarize(results: List[Result]) -> None:
    total = len(results)
    oks = sum(1 for r in results if r.status == "OK")
    errs = total - oks
    print(f"\nSummary: total={total}, OK={oks}, ERROR={errs}")
    # Show some frequent error patterns
    from collections import Counter
    reasons = Counter()
    for r in results:
        if r.status != "OK":
            key = r.message.split("\n")[0][:160]
            reasons[key] += 1
    if reasons:
        print("Top error reasons:")
        for msg, cnt in reasons.most_common(10):
            print(f"  {cnt}x: {msg}")

def extract_missing(msg: str) -> str:
    m = re.search(r"missing=\[([^\]]*)\]", msg)
    if m:
        return "[" + m.group(1) + "]"
    return "N/A"


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe PLETopK kept-count mismatch across diverse inputs")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="compute device")
    parser.add_argument("--seed", type=int, default=123, help="random seed")
    parser.add_argument("--max", type=int, default=0, help="limit number of cases (0 = all)")
    parser.add_argument("--print_errors", action="store_true", help="print detailed error cases")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU")
        args.device = "cpu"

    set_seed(args.seed)
    cases = build_cases()
    if args.max and args.max > 0:
        cases = cases[: args.max]

    results: List[Result] = []
    for i, case in enumerate(cases):
        res = run_case(case, device=args.device)
        results.append(res)
        prefix = "OK   " if res.status == "OK" else "ERROR"
        if res.status == "OK":
            kept_info = f"kept_expected={res.kept_expected}"
        else:
            missing_str = extract_missing(res.message)
            kept_info = f"kept_expected={res.kept_expected}, missing={missing_str}"
        line = (
            f"[{i+1:04d}/{len(cases):04d}] {prefix} "
            f"dist={case.name} B={case.B} N={case.N} C={case.C} r={case.r:.3f} | {kept_info}"
        )
        print(line)
        if res.status != "OK" and args.print_errors:
            print("  ->", res.message)

    summarize(results)


if __name__ == "__main__":
    main()


