import os
import sys
sys.path.append('/home/hoyso/projects/AudioTokenization/BigCodec_SSL')
import time
import argparse
import random
from typing import List, Dict, Any, Optional

import torch
import torch.nn.functional as F
import torchaudio
from torchaudio.transforms import Resample

# Local ToMe implementations
from dtp.tome_ops import ToMeChained, ToMeGreedy, ToMeK2New, OurToMe2, ToMeK2V2, ToPrK2New, ToPrPLETopK, ToPrCPRRTopK, ToPrK2NewChunk

# Prefer existing speaker verification loader (WavLM features via s3prl) if available
def _load_sv_model(device: torch.device, checkpoint_path: str):
    try:
        from speaker_verification.verification import init_model
    except Exception:
        return None
    try:
        model = init_model('wavlm_large', checkpoint_path)
        model.to(device)
        model.eval()
        return model
    except Exception:
        return None


def _maybe_synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def read_filelist(filelist_path: str) -> List[str]:
    with open(filelist_path, "r") as f:
        paths = [ln.strip() for ln in f if ln.strip()]
    return paths


def load_audio_fixed(path: str, target_sr: int, seconds: int) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if wav.dim() == 2:
        wav = wav[:1, :]  # use first channel
    if sr != target_sr:
        wav = Resample(sr, target_sr)(wav)
    wav = wav[0]  # (T,)
    target_len = target_sr * seconds
    if wav.numel() < target_len:
        wav = F.pad(wav, (0, target_len - wav.numel()))
    else:
        wav = wav[:target_len]
    return wav.contiguous()


@torch.no_grad()
def batch_load_audios(filelist_path: str, num_samples: int, target_sr: int, seconds: int, seed: int) -> torch.Tensor:
    rng = random.Random(seed)
    all_paths = read_filelist(filelist_path)
    if len(all_paths) == 0:
        raise RuntimeError(f"Empty filelist: {filelist_path}")
    if len(all_paths) < num_samples:
        selected = rng.choices(all_paths, k=num_samples)
    else:
        selected = rng.sample(all_paths, k=num_samples)
    wavs = [load_audio_fixed(p, target_sr, seconds) for p in selected]
    batch = torch.stack(wavs, dim=0)  # (B, T)
    return batch


def _load_wavlm(model_name: str = "microsoft/wavlm-large", device: torch.device = torch.device("cpu")):
    try:
        from transformers import AutoFeatureExtractor, WavLMModel
    except Exception as e:
        raise RuntimeError("Transformers is required to run this benchmark. Please install transformers.") from e

    processor = AutoFeatureExtractor.from_pretrained(model_name)
    model = WavLMModel.from_pretrained(model_name).to(device)
    model.eval()
    return processor, model


@torch.no_grad()
def wavs_to_tokens_wavlm(
    wavs: torch.Tensor,
    processor,
    model,
    sample_rate: int,
    device: torch.device,
) -> torch.Tensor:
    # feature extractor expects List[np.ndarray]
    wavs_list = [w.cpu().numpy() for w in wavs]
    proc = processor(wavs_list, sampling_rate=sample_rate, return_tensors="pt", padding=True)
    input_values = proc["input_values"].to(device)
    outputs = model(input_values=input_values, output_hidden_states=False, return_dict=True)
    tokens = outputs.last_hidden_state  # (B, N, C)
    return tokens


def cosine_similarity_mean(a: torch.Tensor, b: torch.Tensor) -> float:
    sim = F.cosine_similarity(a, b, dim=-1)
    return sim.mean().item()


def benchmark_tome_chained(tokens: torch.Tensor, r: float, kernel_size: int, device: torch.device) -> Dict[str, Any]:
    model = ToMeChained(r=r, kernel_size=kernel_size).to(device)
    x = tokens
    _maybe_synchronize(device)
    t0 = time.time()
    with torch.no_grad():
        merge_btree = model._create_merge_btree(x)
        direct_to_root = model._resolve_chains(merge_btree)
        merged_x, avg_sim = model.merge(x, direct_to_root)
        unmerged_x = model.unmerge(merged_x, direct_to_root)
    _maybe_synchronize(device)
    dt = (time.time() - t0) * 1000.0
    cos = cosine_similarity_mean(unmerged_x, tokens)
    return {
        "method": "ToMeChained",
        "num_iterations": None,
        "runtime_ms": dt,
        "avg_sim_mean": avg_sim.mean().item(),
        "cos_sim_unmerged_vs_original": cos,
        "n_before": tokens.shape[1],
        "n_after": merged_x.shape[1],
    }


def benchmark_tome_greedy(tokens: torch.Tensor, r: float, kernel_size: int, device: torch.device) -> Dict[str, Any]:
    model = ToMeGreedy(r=r, kernel_size=kernel_size).to(device)
    x = tokens.clone()
    _maybe_synchronize(device)
    t0 = time.time()
    with torch.no_grad():
        merged_x, btree_map, avg_sim = model.merge(x)
        direct_to_root = model.btree_to_root_map(btree_map)
        unmerged_x = model.unmerge(merged_x, direct_to_root)
    _maybe_synchronize(device)
    dt = (time.time() - t0) * 1000.0
    cos = cosine_similarity_mean(unmerged_x, tokens)
    return {
        "method": "ToMeGreedy",
        "num_iterations": None,
        "runtime_ms": dt,
        "avg_sim_mean": avg_sim.mean().item(),
        "cos_sim_unmerged_vs_original": cos,
        "n_before": tokens.shape[1],
        "n_after": merged_x.shape[1],
    }


def benchmark_tome_k2new(tokens: torch.Tensor, r: float, num_iterations: int, device: torch.device) -> Dict[str, Any]:
    model = ToMeK2New(r=r, num_iterations=num_iterations).to(device)
    x = tokens.clone()
    _maybe_synchronize(device)
    t0 = time.time()
    with torch.no_grad():
        merged_x, btree_map, avg_sim = model.compute_merge(x)
        direct_to_root = model.btree_to_root_map(btree_map)
        unmerged_x = model.unmerge(merged_x, direct_to_root)
    _maybe_synchronize(device)
    dt = (time.time() - t0) * 1000.0
    cos = cosine_similarity_mean(unmerged_x, tokens)
    return {
        "method": "ToMeK2New",
        "num_iterations": num_iterations,
        "runtime_ms": dt,
        "avg_sim_mean": avg_sim.mean().item(),
        "cos_sim_unmerged_vs_original": cos,
        "n_before": tokens.shape[1],
        "n_after": merged_x.shape[1],
    }


# def benchmark_tome_k2v2(tokens: torch.Tensor, r: float, num_iterations: int, device: torch.device) -> Dict[str, Any]:
#     model = ToMeK2V2(r=r, num_iterations=num_iterations).to(device)
#     x = tokens.clone()
#     _maybe_synchronize(device)
#     t0 = time.time()
#     with torch.no_grad():
#         merged_x, btree_map, avg_sim = model.compute_merge(x)
#         direct_to_root = model.btree_to_root_map(btree_map)
#         unmerged_x = model.unmerge(merged_x, direct_to_root)
#     _maybe_synchronize(device)
#     dt = (time.time() - t0) * 1000.0
#     cos = cosine_similarity_mean(unmerged_x, tokens)
#     return {
#         "method": "ToMeK2V2",
#         "num_iterations": num_iterations,
#         "runtime_ms": dt,
#         "avg_sim_mean": avg_sim.mean().item(),
#         "cos_sim_unmerged_vs_original": cos,
#         "n_before": tokens.shape[1],
#         "n_after": merged_x.shape[1],
#     }


def benchmark_our_tome2(tokens: torch.Tensor, r: float, num_iterations: int, device: torch.device) -> Dict[str, Any]:
    model = OurToMe2(r=r, num_iterations=num_iterations).to(device)
    x = tokens.clone()
    _maybe_synchronize(device)
    t0 = time.time()
    with torch.no_grad():
        merged_x, btree_map, avg_sim = model.compute_merge(x)
        direct_to_root = model.btree_to_root_map(btree_map)
        unmerged_x = model.unmerge(merged_x, direct_to_root)
    _maybe_synchronize(device)
    dt = (time.time() - t0) * 1000.0
    cos = cosine_similarity_mean(unmerged_x, tokens)
    return {
        "method": "OurToMe2",
        "num_iterations": num_iterations,
        "runtime_ms": dt,
        "avg_sim_mean": avg_sim.mean().item(),
        "cos_sim_unmerged_vs_original": cos,
        "n_before": tokens.shape[1],
        "n_after": merged_x.shape[1],
    }


# def benchmark_our_tome3(tokens: torch.Tensor, r: float, num_iterations: int, device: torch.device) -> Dict[str, Any]:
#     model = OurToMe3(r=r, num_iterations=num_iterations).to(device)
#     x = tokens.clone()
#     _maybe_synchronize(device)
#     t0 = time.time()
#     with torch.no_grad():
#         merged_x, btree_map, avg_sim = model.compute_merge(x)
#         direct_to_root = model.btree_to_root_map(btree_map)
#         unmerged_x = model.unmerge(merged_x, direct_to_root)
#     _maybe_synchronize(device)
#     dt = (time.time() - t0) * 1000.0
#     cos = cosine_similarity_mean(unmerged_x, tokens)
#     return {
#         "method": "OurToMe3",
#         "num_iterations": num_iterations,
#         "runtime_ms": dt,
#         "avg_sim_mean": avg_sim.mean().item(),
#         "cos_sim_unmerged_vs_original": cos,
#         "n_before": tokens.shape[1],
#         "n_after": merged_x.shape[1],
#     }
def benchmark_topr_k2new(tokens: torch.Tensor, r: float, num_iterations: int, device: torch.device) -> Dict[str, Any]:
    model = ToPrK2New(r=r, num_iterations=num_iterations).to(device)
    x = tokens.clone()
    _maybe_synchronize(device)
    t0 = time.time()
    with torch.no_grad():
        merged_x, btree_map, avg_sim = model.compute_merge(x)
        direct_to_root = model.btree_to_root_map(btree_map)
        unmerged_x = model.unmerge(merged_x, direct_to_root)
    _maybe_synchronize(device)
    dt = (time.time() - t0) * 1000.0
    cos = cosine_similarity_mean(unmerged_x, tokens)
    return {
        "method": "ToPrK2New",
        "num_iterations": num_iterations,
        "runtime_ms": dt,
        "avg_sim_mean": avg_sim.mean().item(),
        "cos_sim_unmerged_vs_original": cos,
        "n_before": tokens.shape[1],
        "n_after": merged_x.shape[1],
    }


def benchmark_topr_k2chunk(tokens: torch.Tensor, r: float, num_iterations: int, chunk_size: int, device: torch.device) -> Dict[str, Any]:
    model = ToPrK2NewChunk(r=r, num_iterations=num_iterations, chunk_size=chunk_size).to(device)
    x = tokens.clone()
    _maybe_synchronize(device)
    t0 = time.time()
    with torch.no_grad():
        merged_x, btree_map, avg_sim = model.compute_merge(x)
        direct_to_root = model.btree_to_root_map(btree_map)
        unmerged_x = model.unmerge(merged_x, direct_to_root)
    _maybe_synchronize(device)
    dt = (time.time() - t0) * 1000.0
    cos = cosine_similarity_mean(unmerged_x, tokens)
    return {
        "method": "ToPrK2NewChunk",
        "num_iterations": num_iterations,
        "runtime_ms": dt,
        "avg_sim_mean": avg_sim.mean().item(),
        "cos_sim_unmerged_vs_original": cos,
        "n_before": tokens.shape[1],
        "n_after": merged_x.shape[1],
        "chunk_size": chunk_size,
    }


def benchmark_topr_ple(tokens: torch.Tensor, r: float, beta: float, device: torch.device) -> Dict[str, Any]:
    model = ToPrPLETopK(r=r, beta=beta, use_bin_argmax=True).to(device)
    x = tokens.clone()
    _maybe_synchronize(device)
    t0 = time.time()
    with torch.no_grad():
        merged_x, btree_map, avg_sim = model.compute_merge(x)
        direct_to_root = model.btree_to_root_map(btree_map)
        unmerged_x = model.unmerge(merged_x, direct_to_root)
    _maybe_synchronize(device)
    dt = (time.time() - t0) * 1000.0
    cos = cosine_similarity_mean(unmerged_x, tokens)
    return {
        "method": "ToPrPLETopK",
        "num_iterations": None,
        "runtime_ms": dt,
        "avg_sim_mean": avg_sim.mean().item(),
        "cos_sim_unmerged_vs_original": cos,
        "n_before": tokens.shape[1],
        "n_after": merged_x.shape[1],
    }


def benchmark_topr_cprr(tokens: torch.Tensor, r: float, beta: float, bins: int, device: torch.device) -> Dict[str, Any]:
    model = ToPrCPRRTopK(r=r, beta=beta, bins=bins).to(device)
    x = tokens.clone()
    _maybe_synchronize(device)
    t0 = time.time()
    with torch.no_grad():
        merged_x, btree_map, avg_sim = model.compute_merge(x)
        direct_to_root = model.btree_to_root_map(btree_map)
        unmerged_x = model.unmerge(merged_x, direct_to_root)
    _maybe_synchronize(device)
    dt = (time.time() - t0) * 1000.0
    cos = cosine_similarity_mean(unmerged_x, tokens)
    return {
        "method": "ToPrCPRRTopK",
        "num_iterations": bins,
        "runtime_ms": dt,
        "avg_sim_mean": avg_sim.mean().item(),
        "cos_sim_unmerged_vs_original": cos,
        "n_before": tokens.shape[1],
        "n_after": merged_x.shape[1],
    }


def run_benchmark(
    filelist: str,
    device_str: str = "cpu",
    sample_rate: int = 16000,
    seconds: int = 4,
    batch_size: int = 64,
    r: float = 0.5,
    iterations_list: Optional[List[int]] = None,
    hf_model_name: str = "microsoft/wavlm-large",
    use_sv_loader: bool = True,
    sv_checkpoint: str = "/home/hoyso/projects/AudioTokenization/BigCodec_SSL/wavlm_large_finetune.pth",
    seed: int = 1337,
    chunk_size: int = 50,
) -> List[Dict[str, Any]]:
    if iterations_list is None:
        iterations_list = [2, 4, 8, 16]
    device = torch.device(device_str)

    wavs = batch_load_audios(filelist, batch_size, sample_rate, seconds, seed)
    wavs = wavs.to(device)

    tokens = None
    if use_sv_loader:
        sv_model = _load_sv_model(device, sv_checkpoint)
        if sv_model is not None:
            with torch.no_grad():
                feats = sv_model.get_feat(wavs)
            # feats: (B, C, T) -> tokens: (B, T, C)
            tokens = feats.transpose(1, 2).contiguous()
    if tokens is None:
        processor, wavlm_model = _load_wavlm(hf_model_name, device)
        with torch.no_grad():
            tokens = wavs_to_tokens_wavlm(wavs, processor, wavlm_model, sample_rate, device)  # (B, N, C)
    tokens = tokens.to(device)

    results: List[Dict[str, Any]] = []

    # Methods without num_iterations
    results.append(benchmark_tome_chained(tokens, r=r, kernel_size=2, device=device))
    results.append(benchmark_tome_greedy(tokens, r=r, kernel_size=2, device=device))
    results.append(benchmark_topr_ple(tokens, r=r, beta=1.0, device=device))

    # Methods with num_iterations
    for iters in iterations_list:
        results.append(benchmark_tome_k2new(tokens, r=r, num_iterations=iters, device=device))
        # results.append(benchmark_tome_k2v2(tokens, r=r, num_iterations=iters, device=device))
        results.append(benchmark_our_tome2(tokens, r=r, num_iterations=iters, device=device))
        # results.append(benchmark_our_tome3(tokens, r=r, num_iterations=iters, device=device))
        results.append(benchmark_topr_k2new(tokens, r=r, num_iterations=iters, device=device))
        results.append(benchmark_topr_cprr(tokens, r=r, beta=1.0, bins=max(1, int((r * tokens.shape[1]) ** 0.5)), device=device))
        results.append(benchmark_topr_k2chunk(tokens, r=r, num_iterations=iters, chunk_size=chunk_size, device=device))

    return results


def print_results(results: List[Dict[str, Any]]) -> None:
    headers = [
        "method",
        "num_iterations",
        "n_before",
        "n_after",
        "runtime_ms",
        "avg_sim_mean",
        "cos_sim_unmerged_vs_original",
        "chunk_size",
    ]
    print("\n=== ToMe Benchmark Results ===")
    print("\t".join(headers))
    for r in results:
        row = [
            str(r["method"]),
            str(r["num_iterations"]) if r["num_iterations"] is not None else "-",
            str(r["n_before"]),
            str(r["n_after"]),
            f"{r['runtime_ms']:.2f}",
            f"{r['avg_sim_mean']:.6f}",
            f"{r['cos_sim_unmerged_vs_original']:.6f}",
            str(r.get("chunk_size", "-")),
        ]
        print("\t".join(row))


def save_csv(results: List[Dict[str, Any]], out_path: str) -> None:
    import csv

    headers = [
        "method",
        "num_iterations",
        "n_before",
        "n_after",
        "runtime_ms",
        "avg_sim_mean",
        "cos_sim_unmerged_vs_original",
        "chunk_size",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in results:
            row = r.copy()
            if row.get("num_iterations") is None:
                row["num_iterations"] = ""
            if "chunk_size" not in row:
                row["chunk_size"] = ""
            writer.writerow(row)
    print(f"Saved results to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark ToMe variants on WavLM tokens.")
    parser.add_argument("--filelist", type=str, default="/home/hoyso/projects/AudioTokenization/BigCodec_SSL/filelists/librispeech_test_clean.txt")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda[:id]")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--seconds", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--r", type=float, default=0.5)
    parser.add_argument("--iters", type=int, nargs="*", default=[2, 4, 8, 16])
    parser.add_argument("--hf_model", type=str, default="microsoft/wavlm-large")
    parser.add_argument("--use_sv", action="store_true", help="Use speaker_verification WavLM loader first")
    parser.add_argument("--sv_checkpoint", type=str, default="/home/hoyso/projects/AudioTokenization/BigCodec_SSL/wavlm_large_finetune.pth")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out_csv", type=str, default="")
    parser.add_argument("--chunk_size", type=int, default=50, help="Chunk size for ToPrK2NewChunk")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_benchmark(
        filelist=args.filelist,
        device_str=args.device,
        sample_rate=args.sample_rate,
        seconds=args.seconds,
        batch_size=args.batch_size,
        r=args.r,
        iterations_list=args.iters,
        hf_model_name=args.hf_model,
        use_sv_loader=args.use_sv,
        sv_checkpoint=args.sv_checkpoint,
        seed=args.seed,
        chunk_size=args.chunk_size,
    )
    print_results(results)
    if args.out_csv:
        save_csv(results, args.out_csv)


if __name__ == "__main__":
    main()

# sys.path.append('/home/hoyso/projects/AudioTokenization/BigCodec_SSL/speaker_verification')    # We use wavlm_large_finetune as a vadidation metric during training, https://github.com/microsoft/UniSpeech/tree/main/downstreams/speaker_verification
# from verification import init_model
# model_spk = init_model('wavlm_large','./wavlm_large_finetune.pth')
# model_spk.eval()
# from dtp.tome_ops import ToMeK2New, OurToMe2, ToMeTopK, ToMeGreedy, ToMeChained

# def speaker_verification(wav1, wav2):
#     wav1 = wav1.squeeze(1) # (B, T)
#     wav2 = wav2.squeeze(1) # (B, T)
#     emb1 = model_spk(wav1)
#     emb2 = model_spk(wav2)
#     return torch.nn.functional.cosine_similarity(emb1, emb2, dim=1)


