import time
import torch
from .tome_ops import OurToMe2, OurToMeK


def bench_once(fn, warmup=2, iters=10):
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    end = time.perf_counter()
    return (end - start) / iters


def run_bench(device="cpu", group_sizes=(2, 3, 4)):
    torch.manual_seed(0)
    cases = [
        (1, 256, 64),
        (2, 512, 128),
        (4, 1024, 128),
    ]
    ratios = [0.1, 0.25, 0.5]
    iters = [1, 2, 4]

    print("== OurToMe2 vs OurToMeK ==")
    for (B, N, C) in cases:
        x = torch.randn(B, N, C, device=device)
        for r in ratios:
            for it in iters:
                our2 = OurToMe2(r=r, num_iterations=it)
                t2 = bench_once(lambda: our2.merge(x.clone()), warmup=2, iters=10)
                print(f"OurToMe2 B{B} N{N} C{C} r{r} it{it}: {t2*1e3:.2f}ms")
                for g in group_sizes:
                    ourk = OurToMeK(r=r, num_iterations=it, group_size=g)
                    tk = bench_once(lambda: ourk.merge(x.clone()), warmup=2, iters=10)
                    speedup = t2 / tk if tk > 0 else float('inf')
                    print(f"OurToMeK[g={g}] B{B} N{N} C{C} r{r} it{it}: {tk*1e3:.2f}ms, vs OurToMe2 x{speedup:.2f}")


if __name__ == "__main__":
    print("Device: cpu")
    run_bench(device="cpu")
    if torch.cuda.is_available():
        print("\nDevice: cuda")
        run_bench(device="cuda")


