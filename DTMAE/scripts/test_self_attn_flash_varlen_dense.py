import argparse
import math
import time
import torch

from vq.module import SelfAttention


def _fmt_mem(bytes_val: int) -> str:
    mib = bytes_val / (1024 ** 2)
    if mib < 1024:
        return f"{mib:.1f} MiB"
    gib = mib / 1024
    return f"{gib:.2f} GiB"


def _get_attn_class(impl: str):
    if impl == "flash":
        return SelfAttention
    else:
        raise ValueError(f"Unknown impl: {impl}")


def build_dense_and_nested_inputs(batch_size: int, max_len: int, dim: int, n_heads: int, dtype: torch.dtype, device: torch.device, equal_lengths: bool = False):
    head_dim = dim // n_heads
    assert head_dim * n_heads == dim, "dim must be divisible by n_heads"

    if equal_lengths:
        lengths = torch.full((batch_size,), max_len, device=device, dtype=torch.int64)
    else:
        lengths = torch.randint(low=max(2, max_len // 2), high=max_len + 1, size=(batch_size,), device=device)
        lengths = torch.clamp(lengths, min=2).to(torch.int64)

    x_dense = torch.zeros((batch_size, max_len, dim), device=device, dtype=dtype)
    for b in range(batch_size):
        L = int(lengths[b].item())
        x_dense[b, :L].normal_()

    x_nested_list = [x_dense[b, : int(lengths[b].item())] for b in range(batch_size)]
    x_nested = torch.nested.as_nested_tensor(x_nested_list, layout=torch.jagged)

    pos_ids = torch.arange(max_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, max_len)
    return x_dense, x_nested, lengths, pos_ids


@torch.no_grad()
def test_equivalence(dim: int = 512, n_heads: int = 8, batch_size: int = 4, max_len: int = 256, dtype: str = "half", device: str = "cuda", impl: str = "flash"):
    device = torch.device(device)
    dtype_t = torch.float16 if dtype == "half" else torch.bfloat16

    Attn = _get_attn_class(impl)
    attn_dense = Attn(dim=dim, n_head=n_heads, dropout=0.0, causal=False).to(device=device, dtype=dtype_t)
    attn_varlen = Attn(dim=dim, n_head=n_heads, dropout=0.0, causal=False).to(device=device, dtype=dtype_t)
    attn_varlen.load_state_dict(attn_dense.state_dict())
    attn_dense.eval(); attn_varlen.eval()

    # Enforce equal lengths to make attention context identical between dense and varlen
    x_dense, x_nested, lengths, pos_ids = build_dense_and_nested_inputs(batch_size, max_len, dim, n_heads, dtype_t, device, equal_lengths=True)

    y_dense = attn_dense(x_dense, position_ids=pos_ids)  # [B, T, C]

    # Varlen path: pack and use cu_seqlens/max_seqlen to avoid NestedTensor in module
    x_list = [x_dense[b, :max_len] for b in range(batch_size)]
    x_packed = torch.cat(x_list, dim=0)
    cu_seqlens = torch.cat([torch.zeros(1, dtype=torch.int32, device=device), torch.full((batch_size,), max_len, dtype=torch.int32, device=device).cumsum(0)], dim=0)
    max_seqlen = max_len
    y_varlen_packed = attn_varlen(x_packed, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)  # [B*T, C]
    y_varlen_padded = y_varlen_packed.view(batch_size, max_len, dim)

    max_diff = 0.0
    diffs = []
    for b in range(batch_size):
        L = int(lengths[b].item())
        d = (y_dense[b, :L] - y_varlen_padded[b, :L]).abs()
        diff = d.max().item()
        diffs.append(d.mean().item())
        max_diff = max(max_diff, diff)

    avg_diff = sum(diffs) / len(diffs)
    print(f"Equivalence diffs: max {max_diff:.6f}, mean {avg_diff:.6f}")
    # Tighten tolerance since sequences are equal length now
    tol = 1e-2 if dtype == "half" else 5e-3
    assert max_diff < tol, f"Mismatch too large: {max_diff} >= {tol}"


def build_shared_varlen_case(batch_size: int, max_len: int, dim: int, dtype: torch.dtype, device: torch.device, pad_ratio: float):
    assert 0.0 <= pad_ratio < 1.0
    min_len = max(2, int((1.0 - pad_ratio) * max_len))
    lengths = torch.randint(low=min_len, high=max_len + 1, size=(batch_size,), device=device)
    lengths = torch.clamp(lengths, min=2).to(torch.int64)

    x_dense = torch.zeros((batch_size, max_len, dim), device=device, dtype=dtype)
    x_list = []
    for b in range(batch_size):
        L = int(lengths[b].item())
        xb = torch.randn(L, dim, device=device, dtype=dtype)
        x_list.append(xb)
        x_dense[b, :L] = xb

    x_nested = torch.nested.as_nested_tensor(x_list, layout=torch.jagged)
    # Packed representation for FlashAttention: concat and build cu_seqlens
    x_packed = torch.cat(x_list, dim=0)
    lengths_i32 = lengths.to(torch.int32)
    cu_seqlens = torch.cat([torch.zeros(1, dtype=torch.int32, device=device), lengths_i32.cumsum(0)], dim=0)
    max_seqlen = int(lengths.max().item())
    pos_ids = torch.arange(max_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, max_len)
    return x_dense, x_nested, x_packed, cu_seqlens, max_seqlen, lengths, pos_ids


def _bench_run(fn, *args, warmup: int = 10, iters: int = 50):
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end)
    return ms / iters


def bench_dense(dim=512, n_heads=8, batch_size=8, seq_len=1024, dtype="half", device="cuda", compile_flag=False, impl: str = "flash"):
    device = torch.device(device)
    dtype_t = torch.float16 if dtype == "half" else torch.bfloat16
    Attn = _get_attn_class(impl)
    attn = Attn(dim=dim, n_head=n_heads, dropout=0.0, causal=False).to(device=device, dtype=dtype_t)
    attn.eval()
    if compile_flag:
        attn = torch.compile(attn)
    x = torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype_t)
    pos_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, seq_len)

    def run(a, inp, pids):
        return a(inp, position_ids=pids)

    ms = _bench_run(run, attn, x, pos_ids)
    toks = batch_size * seq_len
    print(f"Dense {'compiled' if compile_flag else 'eager'}: {ms:.2f} ms/iter, {toks / (ms/1000.):.0f} toks/s")


def bench_varlen(dim=512, n_heads=8, batch_size=8, max_len=1024, dtype="half", device="cuda", compile_flag=False, impl: str = "flash"):
    device = torch.device(device)
    dtype_t = torch.float16 if dtype == "half" else torch.bfloat16
    Attn = _get_attn_class(impl)
    attn = Attn(dim=dim, n_head=n_heads, dropout=0.0, causal=False).to(device=device, dtype=dtype_t)
    attn.eval()
    if compile_flag:
        attn = torch.compile(attn)

    # Build jagged inputs with avg len ~ 0.8 * max_len
    lengths = torch.randint(int(0.6 * max_len), max_len + 1, (batch_size,), device=device)
    xs = [torch.randn(int(L.item()), dim, device=device, dtype=dtype_t) for L in lengths]
    x_nt = torch.nested.as_nested_tensor(xs, layout=torch.jagged)

    def run(a, inp):
        return a(inp)

    ms = _bench_run(run, attn, x_nt)
    toks = int(lengths.sum().item())
    print(f"Varlen {'compiled' if compile_flag else 'eager'}: {ms:.2f} ms/iter, {toks / (ms/1000.):.0f} toks/s")


def bench_compare(dim=512, n_heads=8, batch_size=8, max_len=1024, pad_ratio=0.5, dtype="half", device="cuda", compile_flag=False, impl: str = "flash"):
    device = torch.device(device)
    dtype_t = torch.float16 if dtype == "half" else torch.bfloat16
    x_dense, x_nested, x_packed, cu_seqlens, max_seqlen, lengths, pos_ids = build_shared_varlen_case(batch_size, max_len, dim, dtype_t, device, pad_ratio)
    logical_tokens = int(lengths.sum().item())
    physical_tokens = batch_size * max_len

    Attn = _get_attn_class(impl)
    attn_dense = Attn(dim=dim, n_head=n_heads, dropout=0.0, causal=False).to(device=device, dtype=dtype_t).eval()
    attn_varlen = Attn(dim=dim, n_head=n_heads, dropout=0.0, causal=False).to(device=device, dtype=dtype_t).eval()
    if compile_flag:
        attn_dense = torch.compile(attn_dense)
        attn_varlen = torch.compile(attn_varlen)

    def run_dense(a, inp, pids):
        return a(inp, position_ids=pids)

    def run_varlen(a, inp, cu, mx):
        if impl == "flash":
            return a(inp, cu_seqlens=cu, max_seqlen=mx)
        else:
            return a(inp)

    ms_dense = _bench_run(run_dense, attn_dense, x_dense, pos_ids)
    ms_varlen = _bench_run(run_varlen, attn_varlen, (x_packed if impl == "flash" else x_nested), cu_seqlens, max_seqlen)

    dense_logical_tps = logical_tokens / (ms_dense / 1000.0)
    dense_physical_tps = physical_tokens / (ms_dense / 1000.0)
    varlen_tps = logical_tokens / (ms_varlen / 1000.0)

    label = 'compiled' if compile_flag else 'eager'
    print(f"Compare ({label}, pad_ratio={pad_ratio:.2f}):")
    print(f"  Dense padded: {ms_dense:.2f} ms, logical {dense_logical_tps:.0f} toks/s, physical {dense_physical_tps:.0f} toks/s")
    print(f"  Varlen     : {ms_varlen:.2f} ms, logical {varlen_tps:.0f} toks/s")

    # Peak memory (forward-only), exclude compile by warming once then measuring
    def peak_mem_after_warm(fn, *args):
        torch.cuda.synchronize()
        fn(*args)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        fn(*args)
        torch.cuda.synchronize()
        alloc = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.max_memory_reserved()
        return alloc, reserved

    dense_alloc, dense_reserved = peak_mem_after_warm(run_dense, attn_dense, x_dense, pos_ids)
    varlen_alloc, varlen_reserved = peak_mem_after_warm(run_varlen, attn_varlen, (x_packed if impl == "flash" else x_nested), cu_seqlens, max_seqlen)

    print(f"  Dense peak: alloc {_fmt_mem(dense_alloc)}, reserved {_fmt_mem(dense_reserved)}")
    print(f"  Varlen peak: alloc {_fmt_mem(varlen_alloc)}, reserved {_fmt_mem(varlen_reserved)}")


def _bench_run_bwd(fn, *args, warmup: int = 10, iters: int = 50):
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end)
    return ms / iters


def bench_compare_bwd(dim=512, n_heads=8, batch_size=8, max_len=1024, pad_ratio=0.5, dtype="half", device="cuda", compile_flag=False, impl: str = "flash"):
    device = torch.device(device)
    dtype_t = torch.float16 if dtype == "half" else torch.bfloat16
    x_dense, x_nested, x_packed, cu_seqlens, max_seqlen, lengths, pos_ids = build_shared_varlen_case(batch_size, max_len, dim, dtype_t, device, pad_ratio)
    logical_tokens = int(lengths.sum().item())
    physical_tokens = batch_size * max_len

    Attn = _get_attn_class(impl)
    attn_dense = Attn(dim=dim, n_head=n_heads, dropout=0.0, causal=False).to(device=device, dtype=dtype_t).train()
    attn_varlen = Attn(dim=dim, n_head=n_heads, dropout=0.0, causal=False).to(device=device, dtype=dtype_t).train()
    if compile_flag:
        attn_dense = torch.compile(attn_dense)
        attn_varlen = torch.compile(attn_varlen)

    def run_dense_bwd(a, inp, pids):
        a.zero_grad(set_to_none=True)
        out = a(inp, position_ids=pids)
        loss = out.float().pow(2).mean()
        loss.backward()
        return loss

    def run_varlen_bwd(a, a_inp, B, Tmax, C):
        a.zero_grad(set_to_none=True)
        if impl == "flash":
            out = a(a_inp, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        else:
            out_nt = a(a_inp)  # jagged
            out = torch.nested.to_padded_tensor(out_nt, 0.0, (B, Tmax, C))
        loss = out.float().pow(2).mean()
        loss.backward()
        return loss

    ms_dense = _bench_run_bwd(run_dense_bwd, attn_dense, x_dense, pos_ids)
    ms_varlen = _bench_run_bwd(run_varlen_bwd, attn_varlen, (x_packed if impl == "flash" else x_nested), batch_size, max_len, dim)

    dense_logical_tps = logical_tokens / (ms_dense / 1000.0)
    dense_physical_tps = physical_tokens / (ms_dense / 1000.0)
    varlen_tps = logical_tokens / (ms_varlen / 1000.0)

    label = 'compiled' if compile_flag else 'eager'
    print(f"Compare-BWD ({label}, pad_ratio={pad_ratio:.2f}):")
    print(f"  Dense padded: {ms_dense:.2f} ms, logical {dense_logical_tps:.0f} toks/s, physical {dense_physical_tps:.0f} toks/s")
    print(f"  Varlen     : {ms_varlen:.2f} ms, logical {varlen_tps:.0f} toks/s")

    # Peak memory (forward+backward), exclude compile by warming once then measuring
    def peak_mem_after_warm_bwd(fn, *args):
        torch.cuda.synchronize()
        fn(*args)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        fn(*args)
        torch.cuda.synchronize()
        alloc = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.max_memory_reserved()
        return alloc, reserved

    dense_alloc, dense_reserved = peak_mem_after_warm_bwd(run_dense_bwd, attn_dense, x_dense, pos_ids)
    varlen_alloc, varlen_reserved = peak_mem_after_warm_bwd(run_varlen_bwd, attn_varlen, (x_packed if impl == "flash" else x_nested), batch_size, max_len, dim)

    print(f"  Dense peak (bwd): alloc {_fmt_mem(dense_alloc)}, reserved {_fmt_mem(dense_reserved)}")
    print(f"  Varlen peak (bwd): alloc {_fmt_mem(varlen_alloc)}, reserved {_fmt_mem(varlen_reserved)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="half", choices=["half", "bf16"]) 
    p.add_argument("--equiv", action="store_true")
    p.add_argument("--bench", action="store_true")
    p.add_argument("--bench_bwd", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--pad_ratio", type=float, default=0.5)
    p.add_argument("--impl", type=str, default="flash", choices=["flash", "sdpa"])
    args = p.parse_args()

    assert torch.cuda.is_available(), "CUDA is required for FlashAttention benchmarks/tests"

    if args.equiv:
        test_equivalence(dim=args.dim, n_heads=args.n_heads, batch_size=min(4, args.batch), max_len=min(256, args.seq), dtype=args.dtype, device=args.device, impl=args.impl)

    if args.bench:
        bench_compare(dim=args.dim, n_heads=args.n_heads, batch_size=args.batch, max_len=args.seq, pad_ratio=args.pad_ratio, dtype=args.dtype, device=args.device, compile_flag=False, impl=args.impl)
        if args.compile:
            bench_compare(dim=args.dim, n_heads=args.n_heads, batch_size=args.batch, max_len=args.seq, pad_ratio=args.pad_ratio, dtype=args.dtype, device=args.device, compile_flag=True, impl=args.impl)
    if args.bench_bwd:
        bench_compare_bwd(dim=args.dim, n_heads=args.n_heads, batch_size=args.batch, max_len=args.seq, pad_ratio=args.pad_ratio, dtype=args.dtype, device=args.device, compile_flag=False, impl=args.impl)
        if args.compile:
            bench_compare_bwd(dim=args.dim, n_heads=args.n_heads, batch_size=args.batch, max_len=args.seq, pad_ratio=args.pad_ratio, dtype=args.dtype, device=args.device, compile_flag=True, impl=args.impl)


if __name__ == "__main__":
    main()


