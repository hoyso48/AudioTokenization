import torch
import torch.nn as nn
import math
from typing import Tuple, Callable, List, Any, Optional
import torch.nn.functional as F
from vq.module import SelfAttention, RMSNorm

# class OurToMe(nn.Module):
#     """
#     PyTorch nn.Module처럼 동작하도록 설계된 ToMe 레이어.
#     - 인접 토큰만 병합
#     - 최대 병합 사이즈 'm' 제한
#     - 'iterations'에 걸쳐 점진적 병합
#     - Merge & Unmerge 지원
#     """

#     def __init__(self, r: float, m: int, iterations: int):
#         """
#         알고리즘을 위한 하이퍼파라미터를 초기화합니다.
        
#         Args:
#             r (float): 줄일 토큰의 비율 (0.0 ~ 1.0).
#             m (int): 하나의 토큰이 가질 수 있는 최대 원본 토큰 수.
#             iterations (int): 병합을 수행할 반복 횟수.
#         """
#         super().__init__()
#         if not (0.0 <= r <= 1.0):
#             raise ValueError("r must be between 0.0 and 1.0")
#         if m < 1:
#             raise ValueError("m must be at least 1")
#         if iterations < 1:
#             raise ValueError("iterations must be at least 1")
#         if m == 1 and r > 0:
#             raise ValueError("Cannot merge tokens if m=1 and r > 0")
        
#         self.r = r
#         self.m = m
#         self.iterations = iterations

#     def _validate_runtime_args(self, N: int):
#         """런타임에 결정되는 인자에 대한 유효성을 검사합니다."""
#         if self.m > 1:
#             r_max = 1 - 1 / (2**math.floor(math.log2(self.m)))
#             if self.r > r_max:
#                 raise ValueError(f"r={self.r} is too high for m={self.m}. Max r for this m is {r_max:.4f}")

#         num_total_tokens_to_reduce = int(self.r * N)
#         if num_total_tokens_to_reduce % 2 != 0:
#             raise ValueError(f"r * N ({self.r} * {N} = {self.r*N}) must result in an even number of tokens to reduce.")

#     @staticmethod
#     @torch.no_grad()
#     def _get_merge_indices(
#         metric: torch.Tensor, size: torch.Tensor, m: int, num_pairs_to_merge: int
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         병합할 인접 쌍의 인덱스를 계산합니다. (no_grad 컨텍스트)
#         짝수 인덱스와 홀수 인덱스 쌍의 유사도를 계산하여 Top-K를 선택합니다.
#         """
#         B, N, C = metric.shape
        
#         # 짝수/홀수 토큰 및 사이즈 분리
#         # 토큰 수가 홀수일 경우 마지막 토큰은 고려하지 않음
#         L = N // 2
#         even_indices = torch.arange(0, 2 * L, 2, device=metric.device)
#         odd_indices = torch.arange(1, 2 * L, 2, device=metric.device)

#         even_tokens = metric[:, even_indices]
#         odd_tokens = metric[:, odd_indices]
        
#         even_sizes = size[:, even_indices]
#         odd_sizes = size[:, odd_indices]

#         # 1. 인접 유사도 계산 (코사인 유사도 사용)
#         sim = torch.nn.functional.cosine_similarity(even_tokens, odd_tokens, dim=-1) # (B, L)

#         # 2. 'm' 제약 조건에 따른 마스킹
#         future_size = even_sizes + odd_sizes
#         mask = (future_size > m).squeeze(-1)
#         sim.masked_fill_(mask, -float('inf'))

#         # 3. 병합할 쌍이 부족할 경우 처리
#         num_possible_pairs = (~mask).sum(dim=1)
#         k = min(num_pairs_to_merge, num_possible_pairs.min().item())
#         if k < num_pairs_to_merge:
#              print(f"Warning: Not enough pairs to merge. Merging {k} pairs instead of {num_pairs_to_merge}.")
#         if k == 0:
#             return torch.tensor([], dtype=torch.long, device=metric.device), \
#                    torch.tensor([], dtype=torch.long, device=metric.device)

#         # 4. Top-K 선택
#         _, topk_indices = torch.topk(sim, k=k, dim=1) # (B, k)

#         # topk_indices는 (0, L-1) 범위의 인덱스. 원래 짝수/홀수 인덱스로 변환 필요.
#         merge_even_indices = torch.gather(even_indices.expand(B, -1), 1, topk_indices)
#         merge_odd_indices = torch.gather(odd_indices.expand(B, -1), 1, topk_indices)
        
#         return merge_even_indices, merge_odd_indices


#     @staticmethod
#     def _merge_step(
#         x: torch.Tensor, size: torch.Tensor, merge_even_indices: torch.Tensor, merge_odd_indices: torch.Tensor
#     ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
#         """
#         실제 텐서 병합을 수행합니다. (그래디언트 추적)
#         """
#         B, N, C = x.shape
        
#         # 병합될 토큰을 표시하는 마스크 생성
#         merged_mask = torch.zeros(B, N, dtype=torch.bool, device=x.device)
#         merged_mask.scatter_(1, merge_even_indices, True)
#         merged_mask.scatter_(1, merge_odd_indices, True)
        
#         # 병합되지 않을 토큰들
#         unmerged_indices = (~merged_mask).nonzero(as_tuple=True)[1].view(B, -1)
#         unmerged_tokens = torch.gather(x, 1, unmerged_indices.unsqueeze(-1).expand(-1, -1, C))
#         unmerged_sizes = torch.gather(size, 1, unmerged_indices.unsqueeze(-1))

#         # 병합될 토큰들 (짝수 쪽으로 병합)
#         even_toks = torch.gather(x, 1, merge_even_indices.unsqueeze(-1).expand(-1, -1, C))
#         odd_toks = torch.gather(x, 1, merge_odd_indices.unsqueeze(-1).expand(-1, -1, C))
        
#         even_s = torch.gather(size, 1, merge_even_indices.unsqueeze(-1))
#         odd_s = torch.gather(size, 1, merge_odd_indices.unsqueeze(-1))

#         # 가중 평균으로 병합
#         new_size = even_s + odd_s
#         merged_toks = (even_toks * even_s + odd_toks * odd_s) / new_size

#         # 새로운 텐서 생성
#         # 병합된 토큰은 원래 짝수 인덱스 위치에 들어가고, unmerged 토큰과 합쳐짐
#         # 순서를 유지하기 위해, 모든 토큰을 모은 뒤 정렬
        
#         new_x = torch.cat([unmerged_tokens, merged_toks], dim=1)
#         new_size = torch.cat([unmerged_sizes, new_size], dim=1)
        
#         # 다음 단계와 unmerge를 위한 정보
#         # 병합 후 토큰들의 원래 인덱스를 추적
#         original_indices = torch.cat([
#             unmerged_indices,
#             merge_even_indices # 병합된 토큰은 짝수 인덱스를 대표
#         ], dim=1)
        
#         # 정렬하여 순서 유지
#         perm = original_indices.argsort(dim=1)
#         new_x = torch.gather(new_x, 1, perm.unsqueeze(-1).expand(-1, -1, C))
#         new_size = torch.gather(new_size, 1, perm.unsqueeze(-1))
        
#         # Unmerge를 위한 정보 저장
#         merge_info = {
#             "original_N": N,
#             "merge_even_indices": merge_even_indices,
#             "merge_odd_indices": merge_odd_indices,
#         }
#         # print(merge_info)
#         return new_x, new_size, merge_info

#     @staticmethod
#     def _unmerge_step(y: torch.Tensor, merge_info: dict) -> torch.Tensor:
#         """
#         한 단계의 병합을 되돌립니다.
#         """
#         B, N_merged, C = y.shape
#         original_N = merge_info["original_N"]
#         merge_even_indices = merge_info["merge_even_indices"]
#         merge_odd_indices = merge_info["merge_odd_indices"]

#         # 병합되었던 토큰들을 찾기
#         # 현재 y에서 병합된 토큰들이 어느 위치에 있는지 알아내야 함
        
#         # unmerge를 위한 마스크 생성
#         merged_mask_original = torch.zeros(B, original_N, dtype=torch.bool, device=y.device)
#         merged_mask_original.scatter_(1, merge_even_indices, True)
#         merged_mask_original.scatter_(1, merge_odd_indices, True)
        
#         unmerged_indices_original = (~merged_mask_original).nonzero(as_tuple=True)[1].view(B, -1)
        
#         # 병합 후 시퀀스에서 unmerged 토큰과 merged 토큰의 위치를 알아냄
#         temp_indices = torch.cat([
#             unmerged_indices_original,
#             merge_even_indices
#         ], dim=1)
#         perm = temp_indices.argsort(dim=1)
#         inv_perm = perm.argsort(dim=1) # 역순열

#         # y를 원래 순서(unmerged, merged)로 되돌림
#         y_resorted = torch.gather(y, 1, inv_perm.unsqueeze(-1).expand(-1, -1, C))
        
#         num_unmerged = unmerged_indices_original.shape[1]
#         unmerged_toks = y_resorted[:, :num_unmerged]
#         merged_toks = y_resorted[:, num_unmerged:]
        
#         # 원래 크기의 텐서 생성
#         reconstructed_y = torch.zeros(B, original_N, C, device=y.device, dtype=y.dtype)
        
#         # unmerged 토큰 복원
#         reconstructed_y.scatter_(1, unmerged_indices_original.unsqueeze(-1).expand(-1, -1, C), unmerged_toks)
        
#         # merged 토큰 복원 (짝수, 홀수 위치 모두에 동일 값 복사)
#         reconstructed_y.scatter_(1, merge_even_indices.unsqueeze(-1).expand(-1, -1, C), merged_toks)
#         reconstructed_y.scatter_(1, merge_odd_indices.unsqueeze(-1).expand(-1, -1, C), merged_toks)
        
#         return reconstructed_y

#     def merge(self, metric: torch.Tensor) -> Tuple[torch.Tensor, Callable, torch.Tensor]:
#         """
#         입력 텐서에 대해 병합을 수행합니다.
        
#         Args:
#             metric (torch.Tensor): 병합할 토큰 텐서 (B, N, C).
            
#         Returns:
#             - merged_tensor (torch.Tensor): 최종 병합된 텐서.
#             - unmerge_fn (Callable): 병합을 되돌리는 함수.
#             - final_size (torch.Tensor): 병합된 각 토큰의 최종 size.
#         """
#         B, N, C = metric.shape
#         device = metric.device

#         # 런타임 유효성 검사
#         self._validate_runtime_args(N)

#         num_total_pairs_to_merge = int(self.r * N)

#         if num_total_pairs_to_merge == 0:
#             size = torch.ones(B, N, 1, device=device)
#             return metric, lambda x: x, size

#         # 각 이터레이션에 병합할 쌍의 수 분배
#         pairs_per_iter = [num_total_pairs_to_merge // self.iterations] * self.iterations
#         for i in range(num_total_pairs_to_merge % self.iterations):
#             pairs_per_iter[i] += 1

#         x = metric
#         size = torch.ones(B, N, 1, device=device)
#         merge_history: List[dict] = []

#         for i in range(self.iterations):
#             num_pairs = pairs_per_iter[i]
#             if num_pairs == 0 or x.shape[1] < 2:
#                 continue

#             merge_even_indices, merge_odd_indices = self._get_merge_indices(x, size, self.m, num_pairs)
            
#             if merge_even_indices.numel() == 0:
#                 break

#             x, size, merge_info = self._merge_step(x, size, merge_even_indices, merge_odd_indices)
#             merge_history.append(merge_info)

#         def unmerge_fn(y: torch.Tensor) -> torch.Tensor:
#             for merge_info in reversed(merge_history):
#                 y = self._unmerge_step(y, merge_info)
#             return y

#         return x, unmerge_fn, size

#     def forward(self, metric: torch.Tensor) -> Tuple[torch.Tensor, Callable, torch.Tensor]:
#         """nn.Module의 forward와 동일한 역할을 합니다."""
#         return self.merge(metric)

class OurToMe(nn.Module):
    """
    PyTorch nn.Module처럼 동작하도록 설계된 ToMe 레이어.
    - 인접 토큰만 병합
    - 최대 병합 사이즈 'm' 제한
    - 'iterations'에 걸쳐 점진적 병합
    - 학습 시 Gumbel-Top-K 샘플링, 평가 시 결정론적 Top-K 사용
    """

    def __init__(self, r: float, m: Optional[int] = None, iterations: int = 1, tau: float = 0.1):
        """
        알고리즘을 위한 하이퍼파라미터를 초기화합니다.
        
        Args:
            r (float): 줄일 토큰의 비율 (0.0 ~ 1.0).
            m (int): 하나의 토큰이 가질 수 있는 최대 원본 토큰 수.
            iterations (int): 병합을 수행할 반복 횟수.
            tau (float): Gumbel-Top-K 샘플링을 위한 temperature.
        """
        super().__init__()
        if not (0.0 <= r <= 1.0):
            raise ValueError("r must be between 0.0 and 1.0")
        if m is not None and m < 1:
            raise ValueError("m must be at least 1")
        if iterations < 1:
            raise ValueError("iterations must be at least 1")
        if m is not None and m == 1 and r > 0:
            raise ValueError("Cannot merge tokens if m=1 and r > 0")
        
        self.r = r
        self.m = m
        self.iterations = iterations
        self.tau = tau

    def _validate_runtime_args(self, N: int):
        """런타임에 결정되는 인자에 대한 유효성을 검사합니다."""
        if self.m is not None and self.m > 1:
            r_max = 1 - 1 / (2**math.floor(math.log2(self.m)))
            if self.r > r_max:
                raise ValueError(f"r={self.r} is too high for m={self.m}. Max r for this m is {r_max:.4f}")

        num_total_tokens_to_reduce = int(self.r * N)
        if num_total_tokens_to_reduce % 2 != 0:
            raise ValueError(f"r * N ({self.r} * {N} = {self.r*N}) must result in an even number of tokens to reduce.")

    # @staticmethod 제거하고 self를 인자로 받도록 수정
    @torch.no_grad()
    def _get_merge_indices(
        self, metric: torch.Tensor, size: torch.Tensor, m: int, num_pairs_to_merge: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        병합할 인접 쌍의 인덱스를 계산합니다.
        학습 시에는 Gumbel-Top-K 샘플링, 평가 시에는 일반 Top-K를 사용합니다.
        """
        B, N, C = metric.shape
        device = metric.device
        
        L = N // 2
        even_indices = torch.arange(0, 2 * L, 2, device=device)
        odd_indices = torch.arange(1, 2 * L, 2, device=device)

        even_tokens = metric[:, even_indices]
        odd_tokens = metric[:, odd_indices]
        even_sizes = size[:, even_indices]
        odd_sizes = size[:, odd_indices]

        sim = torch.nn.functional.cosine_similarity(even_tokens, odd_tokens, dim=-1)

        future_size = even_sizes + odd_sizes
        if m is not None:
            mask = (future_size > m).squeeze(-1)
        else:
            mask = torch.zeros(B, L, dtype=torch.bool, device=device)
        sim.masked_fill_(mask, -float('inf'))

        num_possible_pairs = (~mask).sum(dim=1)
        k = min(num_pairs_to_merge, num_possible_pairs.min().item())
        if k < num_pairs_to_merge and self.training:
             print(f"Warning: Merging {k} pairs instead of {num_pairs_to_merge}.")
        if k == 0:
            return torch.tensor([], dtype=torch.long, device=device), \
                   torch.tensor([], dtype=torch.long, device=device)

        # if self.training:
        #     gumbel_noise = -torch.log(-torch.log(torch.rand_like(sim) + 1e-9) + 1e-9)
        #     sampled_scores = (sim / self.tau) + gumbel_noise
        #     _, topk_indices = torch.topk(sampled_scores, k=k, dim=1)
        # else:
        #     _, topk_indices = torch.topk(sim, k=k, dim=1)
        
        _, topk_indices = torch.topk(sim, k=k, dim=1)
        
        merge_even_indices = torch.gather(even_indices.expand(B, -1), 1, topk_indices)
        merge_odd_indices = torch.gather(odd_indices.expand(B, -1), 1, topk_indices)
        
        return merge_even_indices, merge_odd_indices

    @staticmethod
    def _merge_step(
        x: torch.Tensor, size: torch.Tensor, merge_even_indices: torch.Tensor, merge_odd_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        B, N, C = x.shape
        merged_mask = torch.zeros(B, N, dtype=torch.bool, device=x.device)
        merged_mask.scatter_(1, merge_even_indices, True)
        merged_mask.scatter_(1, merge_odd_indices, True)
        unmerged_indices = (~merged_mask).nonzero(as_tuple=True)[1].view(B, -1)
        unmerged_tokens = torch.gather(x, 1, unmerged_indices.unsqueeze(-1).expand(-1, -1, C))
        unmerged_sizes = torch.gather(size, 1, unmerged_indices.unsqueeze(-1))
        even_toks = torch.gather(x, 1, merge_even_indices.unsqueeze(-1).expand(-1, -1, C))
        odd_toks = torch.gather(x, 1, merge_odd_indices.unsqueeze(-1).expand(-1, -1, C))
        even_s = torch.gather(size, 1, merge_even_indices.unsqueeze(-1))
        odd_s = torch.gather(size, 1, merge_odd_indices.unsqueeze(-1))
        new_size = even_s + odd_s
        merged_toks = (even_toks * even_s + odd_toks * odd_s) / new_size
        new_x_cat = torch.cat([unmerged_tokens, merged_toks], dim=1)
        new_size_cat = torch.cat([unmerged_sizes, new_size], dim=1)
        original_indices = torch.cat([unmerged_indices, merge_even_indices], dim=1)
        perm = original_indices.argsort(dim=1)
        new_x = torch.gather(new_x_cat, 1, perm.unsqueeze(-1).expand(-1, -1, C))
        new_size = torch.gather(new_size_cat, 1, perm.unsqueeze(-1))
        merge_info = {"original_N": N, "merge_even_indices": merge_even_indices, "merge_odd_indices": merge_odd_indices}
        return new_x, new_size, merge_info

    @staticmethod
    def _unmerge_step(y: torch.Tensor, merge_info: dict) -> torch.Tensor:
        B, N_merged, C = y.shape
        original_N = merge_info["original_N"]
        merge_even_indices = merge_info["merge_even_indices"]
        merge_odd_indices = merge_info["merge_odd_indices"]
        merged_mask_original = torch.zeros(B, original_N, dtype=torch.bool, device=y.device)
        merged_mask_original.scatter_(1, merge_even_indices, True)
        merged_mask_original.scatter_(1, merge_odd_indices, True)
        unmerged_indices_original = (~merged_mask_original).nonzero(as_tuple=True)[1].view(B, -1)
        temp_indices = torch.cat([unmerged_indices_original, merge_even_indices], dim=1)
        perm = temp_indices.argsort(dim=1)
        inv_perm = perm.argsort(dim=1)
        y_resorted = torch.gather(y, 1, inv_perm.unsqueeze(-1).expand(-1, -1, C))
        num_unmerged = unmerged_indices_original.shape[1]
        unmerged_toks = y_resorted[:, :num_unmerged]
        merged_toks = y_resorted[:, num_unmerged:]
        reconstructed_y = torch.zeros(B, original_N, C, device=y.device, dtype=y.dtype)
        reconstructed_y.scatter_(1, unmerged_indices_original.unsqueeze(-1).expand(-1, -1, C), unmerged_toks)
        reconstructed_y.scatter_(1, merge_even_indices.unsqueeze(-1).expand(-1, -1, C), merged_toks)
        reconstructed_y.scatter_(1, merge_odd_indices.unsqueeze(-1).expand(-1, -1, C), merged_toks)
        return reconstructed_y

    def merge(self, metric: torch.Tensor) -> Tuple[torch.Tensor, Callable, torch.Tensor]:
        B, N, C = metric.shape
        device = metric.device

        self._validate_runtime_args(N)

        num_total_pairs_to_merge = int(self.r * N)

        if num_total_pairs_to_merge == 0:
            size = torch.ones(B, N, 1, device=device)
            return metric, lambda x: x, size

        pairs_per_iter = [num_total_pairs_to_merge // self.iterations] * self.iterations
        for i in range(num_total_pairs_to_merge % self.iterations):
            pairs_per_iter[i] += 1

        x = metric
        size = torch.ones(B, N, 1, device=device)
        merge_history: List[dict] = []

        for i in range(self.iterations):
            num_pairs = pairs_per_iter[i]
            if num_pairs == 0 or x.shape[1] < 2:
                continue

            merge_even_indices, merge_odd_indices = self._get_merge_indices(x, size, self.m, num_pairs)
            
            if merge_even_indices.numel() == 0:
                break

            x, size, merge_info = self._merge_step(x, size, merge_even_indices, merge_odd_indices)
            merge_history.append(merge_info)

        def unmerge_fn(y: torch.Tensor) -> torch.Tensor:
            for merge_info in reversed(merge_history):
                y = self._unmerge_step(y, merge_info)
            return y

        return x, unmerge_fn, size

    def forward(self, metric: torch.Tensor) -> Tuple[torch.Tensor, Callable, torch.Tensor]:
        return self.merge(metric)

class OurToMeMaskingUpsampler(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.mask = nn.Parameter(torch.randn(1, 1, dim))

    def forward(self, metric: torch.Tensor, sizes: torch.Tensor):
        """
        Upsamples merged tokens back to original sequence length, filling gaps with a mask token.

        Args:
            metric (torch.Tensor): The merged tokens from ToMe, shape [B, N, C].
            sizes (torch.Tensor): The size of each merged token, shape [B, N, 1].

        Returns:
            torch.Tensor: The upsampled tensor of shape [B, orig_N_max, C].
        """
        B, N, C = metric.shape
        
        # Squeeze and ensure integer type for operations
        sizes_squeezed = sizes.squeeze(-1).long()

        # Calculate original length for each item and find the max for padding
        orig_N_per_item = torch.sum(sizes_squeezed, dim=1)
        orig_N_max = torch.max(orig_N_per_item).item()

        # Calculate placement indices (start index of each token's block)
        placement_indices = torch.cumsum(sizes_squeezed, dim=1) - sizes_squeezed

        # Create the output tensor filled with the learnable mask token
        output = self.mask.to(dtype=metric.dtype).expand(B, orig_N_max, C).clone()

        # Expand indices for scatter to match the tensor dimensions
        indices_for_scatter = placement_indices.unsqueeze(-1).expand(B, N, C)
        
        # Scatter the merged tokens into the output tensor
        output.scatter_(dim=1, index=indices_for_scatter, src=metric)

        return output


class GreedyToMe(nn.Module):
    """
    Performs greedy token merging with batch support using a hybrid approach.
    - Similarity calculation is parallelized across the batch.
    - The iterative merging process is handled by a for-loop over the batch items.
    """
    def __init__(self, r: float, m: int):
        super().__init__()
        if not 0 < r < 1:
            raise ValueError("Merge ratio 'r' must be between 0 and 1.")
        if m < 2:
            raise ValueError("Max merge 'm' must be at least 2.")
        self.r = r
        self.m = m

    def _process_single(self, x_single: torch.Tensor, initial_sims: torch.Tensor) -> Tuple[torch.Tensor, Callable, torch.Tensor]:
        """
        Processes a single item from the batch.
        Args:
            x_single (torch.Tensor): A single tensor item of shape (N, C).
            initial_sims (torch.Tensor): Pre-calculated similarities of shape (N-1).
        """
        N, C = x_single.shape
        num_merges = int(self.r * N)
        
        # Clone to avoid modifying the original tensor views
        with torch.no_grad():
            current_x = x_single.clone()
            current_sims = initial_sims.clone()
            sizes = torch.ones(N, 1, device=x_single.device, dtype=x_single.dtype)
            mapping = [[i] for i in range(N)]

        for _ in range(num_merges):
            L = current_x.shape[0]
            if L <= 2: # Need at least 2 tokens to have one similarity value
                break

            valid_mask = (sizes[:-1] + sizes[1:]).squeeze(-1) <= self.m
            
            if not torch.any(valid_mask):
                break

            sims_masked = current_sims.clone()
            sims_masked[~valid_mask] = -torch.inf

            best_idx = torch.argmax(sims_masked).item()

            s1, s2 = sizes[best_idx], sizes[best_idx + 1]
            total_size = s1 + s2
            
            new_token = (current_x[best_idx] * s1 + current_x[best_idx + 1] * s2) / total_size

            # Update the token tensor
            current_x = torch.cat([
                current_x[:best_idx],
                new_token.unsqueeze(0),
                current_x[best_idx + 2:]
            ], dim=0)

            sizes = torch.cat([
                sizes[:best_idx],
                total_size.unsqueeze(0),
                sizes[best_idx + 2:]
            ], dim=0)
            
            merged_map_entry = mapping[best_idx] + mapping[best_idx + 1]
            mapping = mapping[:best_idx] + [merged_map_entry] + mapping[best_idx + 2:]
            
            # --- Similarity Update (Optimization) ---
            # Remove the similarity of the merged pair
            current_sims = torch.cat([current_sims[:best_idx], current_sims[best_idx+1:]])

            # Recalculate similarity for the new token's neighbors
            if best_idx > 0:
                # Similarity with the token to the left
                normed_left = F.normalize(current_x[best_idx-1:best_idx], p=2, dim=1)
                normed_new = F.normalize(current_x[best_idx:best_idx+1], p=2, dim=1)
                current_sims[best_idx-1] = (normed_left * normed_new).sum()
            
            if best_idx < len(current_sims):
                # Similarity with the token to the right
                normed_new = F.normalize(current_x[best_idx:best_idx+1], p=2, dim=1)
                normed_right = F.normalize(current_x[best_idx+1:best_idx+2], p=2, dim=1)
                current_sims[best_idx] = (normed_new * normed_right).sum()


        # Closure for the unmerge function for this specific item
        def unmerge_fn(merged_x_b: torch.Tensor) -> torch.Tensor:
            unmerged_x = torch.zeros(N, C, device=merged_x_b.device, dtype=merged_x_b.dtype)
            for i, original_indices in enumerate(mapping):
                unmerged_x[original_indices, :] = merged_x_b[i, :]
            return unmerged_x
        
        return current_x, unmerge_fn, sizes.squeeze(-1)


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[Callable], torch.Tensor]:
        """
        Applies the greedy merging process to a batch of tensors.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N, C).

        Returns:
            - merged_x_padded (torch.Tensor): The padded tensor after merging,
                                              shape (B, N_merged_max, C).
            - unmerge_fns (List[Callable]): A list of unmerge functions, one for each
                                            item in the batch.
            - final_sizes_padded (torch.Tensor): Padded sizes of the final tokens,
                                                 shape (B, N_merged_max).
        """
        B, N, C = x.shape

        # [Parallel] Calculate initial similarities for the whole batch
        with torch.no_grad():
            normed_x = F.normalize(x, p=2, dim=2)
            initial_sims = (normed_x[:, :-1, :] * normed_x[:, 1:, :]).sum(dim=2)
            initial_sims = initial_sims + torch.randn(initial_sims.shape, device=x.device, dtype=x.dtype) * 0.0001

        # [For-loop] Process each item in the batch
        merged_batch = []
        sizes_batch = []
        unmerge_fns = []

        for i in range(B):
            merged_x, unmerge_fn, sizes = self._process_single(x[i], initial_sims[i])
            merged_batch.append(merged_x)
            sizes_batch.append(sizes)
            unmerge_fns.append(unmerge_fn)
        
        # Pad the sequences to have the same length
        # merged_x_padded = nn.utils.rnn.pad_sequence(merged_batch, batch_first=True, padding_value=0.0)
        # final_sizes_padded = nn.utils.rnn.pad_sequence(sizes_batch, batch_first=True, padding_value=0.0)

        # The unmerge function needs to handle the padded input
        def unmerge_batch_fn(tensor: torch.Tensor) -> torch.Tensor:
            unmerged_list = []
            for i in range(B):
                # Get the original length of the merged sequence for this item
                # original_len = len(merged_batch[i])
                # Unpad the tensor for this item before passing to its specific unmerge function
                # unpadded_tensor = padded_tensor[i, :original_len, :]
                unmerged_list.append(unmerge_fns[i](tensor[i]))
            return torch.stack(unmerged_list, dim=0)

        return torch.stack(merged_batch, dim=0), unmerge_batch_fn, torch.stack(sizes_batch, dim=0)

def mps_gather_workaround(input, dim, index):
    # MPS 디바이스를 위한 gather 워크어라운드
    if input.device.type == "mps" and input.shape[-1] == 1:
        return torch.gather(input.unsqueeze(-1), dim - 1 if dim < 0 else dim, index.unsqueeze(-1)).squeeze(-1)
    else:
        return torch.gather(input, dim, index)

class ToMeSD1D(nn.Module):
    """
    ToMeSD의 로직을 1D 데이터에 맞게 변형하고, 클래스 기반으로 재구성.
    """
    def __init__(self, r: float, kernel_size: int = 4, stride: int = 4, no_rand: bool = False, generator: torch.Generator = None):
        """
        1D ToMeSD 레이어를 초기화합니다.
        
        Args:
            r (int): 각 병합 단계에서 줄일 토큰의 (절대적인) 수.
            kernel_size (int): dst 토큰을 선택할 커널(그룹)의 크기.
            stride (int): 커널을 적용할 간격.
            no_rand (bool): dst 토큰 선택 시 랜덤성을 사용할지 여부. False이면 각 커널의 첫 번째 토큰을 dst로 선택.
            generator (torch.Generator): 랜덤성 제어를 위한 PyTorch Generator.
        """
        super().__init__()
        self.r = r
        self.kernel_size = kernel_size
        self.stride = stride
        self.no_rand = no_rand
        self.generator = generator

    @torch.no_grad()
    def _get_merge_unmerge_fns(self, metric: torch.Tensor) -> Tuple[Callable, Callable]:
        """
        입력 텐서를 기반으로 merge와 unmerge 함수를 생성합니다.
        (배치 차원 불일치 버그 수정)
        """
        B, N, C = metric.shape
        device = metric.device

        r = int(self.r * N)
        if self.r <= 0:
            return lambda x, mode=None: x, lambda x: x

        gather = mps_gather_workaround
        
        num_kernels = N // self.stride
        
        if self.no_rand:
            rand_kernel_idx = torch.zeros(num_kernels, 1, device=device, dtype=torch.int64)
        else:
            rand_kernel_idx = torch.randint(
                self.kernel_size, size=(num_kernels, 1), device=device, generator=self.generator
            )
            
        idx_buffer_view = torch.zeros(num_kernels, self.kernel_size, device=device, dtype=torch.int64)
        idx_buffer_view.scatter_(dim=1, index=rand_kernel_idx, src=-torch.ones_like(rand_kernel_idx))
        
        idx_buffer = torch.zeros(N, device=device, dtype=torch.int64)
        for i in range(num_kernels):
            start = i * self.stride
            end = start + self.kernel_size
            if end <= N:
                idx_buffer[start:end] = idx_buffer_view[i]

        # --- 버그 수정 부분 ---
        # rand_sorted_idx의 배치 차원을 B로 확장
        # reshape(1, -1, 1) -> expand(B, -1, -1)
        rand_sorted_idx = idx_buffer.argsort(dim=0).view(1, -1, 1).expand(B, -1, -1)
        # --- 수정 완료 ---

        del idx_buffer, idx_buffer_view
        
        num_dst = num_kernels
        src_indices = rand_sorted_idx[:, num_dst:, :]
        dst_indices = rand_sorted_idx[:, :num_dst, :]
        
        # 이제 src_indices와 dst_indices의 shape은 (B, num_src, 1), (B, num_dst, 1)이 됨

        def split(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            _B, _N, _C = x.shape
            # expand 불필요, _B가 B와 같으므로
            src = gather(x, dim=1, index=src_indices.expand(-1, -1, _C))
            dst = gather(x, dim=1, index=dst_indices.expand(-1, -1, _C))
            return src, dst

        metric_normed = metric / metric.norm(dim=-1, keepdim=True)
        src_metric, dst_metric = split(metric_normed)
        scores = src_metric @ dst_metric.transpose(-1, -2)

        r_eff = min(r, src_metric.shape[1])

        node_max, node_idx = scores.max(dim=-1)
        # edge_idx도 배치 차원을 B로 가짐
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unmerged_src_idx = edge_idx[..., r_eff:, :]
        merged_src_idx = edge_idx[..., :r_eff, :]
        
        merged_dst_idx = gather(node_idx[..., None], dim=-2, index=merged_src_idx)
        
        def merge(x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
            src_x, dst_x = split(x)
            _B, _t1, _c = src_x.shape

            unmerged_src_x = gather(src_x, dim=-2, index=unmerged_src_idx.expand(-1, -1, _c))
            merged_src_x = gather(src_x, dim=-2, index=merged_src_idx.expand(-1, -1, _c))
            
            if hasattr(torch.Tensor, "scatter_reduce_"):
                 dst_x.scatter_reduce_(-2, merged_dst_idx.expand(-1, -1, _c), merged_src_x, reduce=mode)
            else:
                 dst_x.scatter_add_(-2, merged_dst_idx.expand(-1, -1, _c), merged_src_x)

            return torch.cat([unmerged_src_x, dst_x], dim=1)

        def unmerge(x: torch.Tensor) -> torch.Tensor:
            num_unmerged_src = unmerged_src_idx.shape[1]
            unmerged_part, dst_part = x[..., :num_unmerged_src, :], x[..., num_unmerged_src:, :]
            _B, _, _c = unmerged_part.shape

            merged_src_values = gather(dst_part, dim=-2, index=merged_dst_idx.expand(-1, -1, _c))
            
            out = torch.zeros(B, N, C, device=device, dtype=x.dtype)
            
            # 1. dst 토큰들 채우기
            out.scatter_(1, dst_indices.expand(-1, -1, C), dst_part)
            
            # --- 버그 수정 부분 ---
            # 이제 gather의 self와 index 모두 배치 크기 B를 가짐
            # 2. 병합되지 않았던 src 토큰들 채우기
            unmerged_orig_idx = gather(src_indices.squeeze(-1), 1, unmerged_src_idx.squeeze(-1)).unsqueeze(-1)
            out.scatter_(1, unmerged_orig_idx.expand(-1, -1, C), unmerged_part)
            
            # 3. 병합되었던 src 토큰들 채우기
            merged_orig_idx = gather(src_indices.squeeze(-1), 1, merged_src_idx.squeeze(-1)).unsqueeze(-1)
            out.scatter_(1, merged_orig_idx.expand(-1, -1, C), merged_src_values)
            # --- 수정 완료 ---
            
            return out

        return merge, unmerge


    def forward(self, metric: torch.Tensor) -> Tuple[Callable, Callable]:
        """
        nn.Module의 forward와 동일한 역할을 합니다.
        주어진 텐서에 대한 merge와 unmerge 함수를 반환합니다.
        """
        return self._get_merge_unmerge_fns(metric)


class ToMeSD1DMasking(nn.Module):
    """
    ToMeSD 로직 기반 1D 레이어의 최종 버전.
    - unmerge 시 mask_token 사용
    - merge 결과 정렬
    - 학습/평가 시 다른 병합 전략 사용
    """
    def __init__(self, r: float, kernel_size: int = 4, stride: int = 4, no_rand: bool = False, generator: torch.Generator = None, tau: float = 0.1):
        super().__init__()
        self.r = r
        self.kernel_size = kernel_size
        self.stride = stride
        self.no_rand = no_rand
        self.generator = generator
        self.mask_token = None
        self.tau = tau
    # @staticmethod 제거, self를 인자로 받도록 수정
    @torch.no_grad()
    def _get_bipartite_indices(
        self, B: int, N: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """src와 dst 그룹의 원본 인덱스를 계산합니다."""
        num_kernels = N // self.stride
        
        if self.no_rand:
            rand_kernel_idx = torch.zeros(num_kernels, 1, device=device, dtype=torch.int64)
        else:
            rand_kernel_idx = torch.randint(
                self.kernel_size, size=(num_kernels, 1), device=device, generator=self.generator
            )
            
        idx_buffer_view = torch.zeros(num_kernels, self.kernel_size, device=device, dtype=torch.int64)
        idx_buffer_view.scatter_(dim=1, index=rand_kernel_idx, src=-torch.ones_like(rand_kernel_idx))
        
        idx_buffer = torch.zeros(N, device=device, dtype=torch.int64)
        for i in range(num_kernels):
            start, end = i * self.stride, i * self.stride + self.kernel_size
            if end <= N:
                idx_buffer[start:end] = idx_buffer_view[i]

        rand_sorted_idx = idx_buffer.argsort(dim=0).view(1, -1, 1).expand(B, -1, -1)
        
        num_dst = num_kernels
        src_indices = rand_sorted_idx[:, num_dst:, :]
        dst_indices = rand_sorted_idx[:, :num_dst, :]
        
        return src_indices, dst_indices
        
    def forward(self, metric: torch.Tensor) -> Tuple[Callable, Callable]:
        B, N, C = metric.shape
        device = metric.device
        gather = mps_gather_workaround
        
        # 1. mask_token 초기화 (필요 시)
        if self.mask_token is None or self.mask_token.shape[2] != C:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, C, device=device))
            nn.init.normal_(self.mask_token, std=0.02)
        
        # 2. 병합할 토큰 수 계산
        r_eff = int(self.r * N)
        if r_eff <= 0:
            return (lambda x, mode=None: x), (lambda x: x)
            
        # 3. no_grad 컨텍스트에서 모든 인덱싱 관련 작업 수행
        with torch.no_grad():
            # 3.1. src/dst 파티셔닝
            src_indices, dst_indices = self._get_bipartite_indices(B, N, device)
            
            def split(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                _B, _N, _C = x.shape
                src = gather(x, dim=1, index=src_indices.expand(-1, -1, _C))
                dst = gather(x, dim=1, index=dst_indices.expand(-1, -1, _C))
                return src, dst
                
            # 3.2. 유사도 계산
            metric_normed = metric / (metric.norm(dim=-1, keepdim=True) + 1e-6)
            src_metric, dst_metric = split(metric_normed)
            scores = src_metric @ dst_metric.transpose(-1, -2)
            
            r_eff = min(r_eff, src_metric.shape[1])
            
            # # 3.3. 학습/평가 모드에 따른 Top-K 선택
            # if self.training:
            #     # Gumbel-Top-K 샘플링
            #     gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-9) + 1e-9)
            #     sampled_scores = scores / self.tau + gumbel_noise
            #     node_max, node_idx = sampled_scores.max(dim=-1)
            #     edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
            # else:
            #     # 결정론적 Top-K
            #     node_max, node_idx = scores.max(dim=-1)
            #     edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
            node_max, node_idx = scores.max(dim=-1)
            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
                
            unmerged_src_idx = edge_idx[..., r_eff:, :]
            merged_src_idx = edge_idx[..., :r_eff, :]
            merged_dst_idx = gather(node_idx[..., None], dim=-2, index=merged_src_idx)
            
        # 4. merge와 unmerge_with_mask 함수 정의 (클로저)
        def merge(x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
            src_x, dst_x = split(x)
            _B, _t1, _c = src_x.shape

            unmerged_src_x = gather(src_x, dim=-2, index=unmerged_src_idx.expand(-1, -1, _c))
            merged_src_x = gather(src_x, dim=-2, index=merged_src_idx.expand(-1, -1, _c))

            # PyTorch 1.12 이상에서는 scatter_reduce 사용 가능
            if hasattr(dst_x, "scatter_reduce_"):
                 dst_x.scatter_reduce_(-2, merged_dst_idx.expand(-1, -1, _c), merged_src_x, reduce=mode)
            else:
                 dst_x.scatter_add_(-2, merged_dst_idx.expand(-1, -1, _c), merged_src_x)
            
            y = torch.cat([unmerged_src_x, dst_x], dim=1)
            
            unmerged_original_indices = gather(src_indices, 1, unmerged_src_idx)
            y_original_indices = torch.cat([unmerged_original_indices, dst_indices], dim=1)
            perm = y_original_indices.squeeze(-1).argsort(dim=1)
            y_sorted = gather(y, 1, perm.unsqueeze(-1).expand(-1, -1, _c))
            
            return y_sorted

        def unmerge_with_mask(merged_x: torch.Tensor) -> torch.Tensor:
            _B, _N_merged, _C = merged_x.shape
            
            # 살아남은 토큰들의 원본 인덱스
            unmerged_original_indices = gather(src_indices, 1, unmerged_src_idx)
            surviving_indices = torch.cat([unmerged_original_indices, dst_indices], dim=1)
            
            # 정렬된 merged_x에 맞춰 살아남은 인덱스도 정렬
            perm = surviving_indices.squeeze(-1).argsort(dim=1)
            sorted_surviving_indices = gather(surviving_indices, 1, perm.unsqueeze(-1))

            # 모든 자리를 mask_token으로 초기화
            out = self.mask_token.to(merged_x.dtype).expand(_B, N, _C)
            
            # 살아남은 토큰들을 원래 자리에 scatter
            out = out.scatter(1, sorted_surviving_indices.expand(-1, -1, _C), merged_x)

            return out

        return merge, unmerge_with_mask


import torch
from torch.autograd import Function
import torch.nn.functional as F
import numpy as np
import scipy.special as spec
# import matplotlib.pyplot as plt
# import seaborn as sns
# from IPython.display import clear_output

# =============================================================================
# 1. & 2. SoftTopK 구현 및 래퍼 함수 (이전과 동일)
# =============================================================================
EPS = torch.finfo(torch.float32).tiny
INF = np.finfo(np.float32).max

def softtopk_forward_np(logits, k):
    batchsize, n = logits.shape
    messages = -INF * np.ones((batchsize, n, k + 1))
    messages[:, 0, 0] = 0
    messages[:, 0, 1] = logits[:, 0]
    for i in range(1, n):
        for j in range(k + 1):
            logp_dont_use = messages[:, i - 1, j]
            logp_use = (
                messages[:, i - 1, j - 1] + logits[:, i] if j > 0 else -INF)
            message = np.logaddexp(logp_dont_use, logp_use)
            messages[:, i, j] = message
    return messages

def softtopk_backward_np(logits, k):
    batchsize, n = logits.shape
    messages = -INF * np.ones((batchsize, n, k + 1))
    messages[:, n - 1, k] = 0
    for i in range(n - 2, -1, -1):
        for j in range(k + 1):
            logp_dont_use = messages[:, i + 1, j]
            logp_use = (
                messages[:, i + 1, j + 1] + logits[:, i + 1] if j < k else -INF)
            message = np.logaddexp(logp_dont_use, logp_use)
            messages[:, i, j] = message
    return messages

def softtopk_np(logits, k):
    batchsize = logits.shape[0]
    f = softtopk_forward_np(logits, k)
    b = softtopk_backward_np(logits, k)
    initial_f = -INF * np.ones((batchsize, 1, k + 1))
    initial_f[:, :, 0] = 0
    ff = np.concatenate([initial_f, f[:, :-1, :]], axis=1)
    lse0 = spec.logsumexp(ff + b, axis=2)
    lse1 = spec.logsumexp(ff[:, :, :-1] + b[:, :, 1:], axis=2) + logits
    return np.exp(lse1 - np.logaddexp(lse0, lse1))

class SoftTopK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, k, eps):
        ctx.save_for_backward(logits)
        ctx.k = k
        ctx.eps = eps
        dtype = logits.dtype
        device = logits.device
        mu_np = softtopk_np(logits.cpu().detach().numpy(), k)
        mu = torch.from_numpy(mu_np).type(dtype).to(device)
        return mu

    @staticmethod
    def backward(ctx, grad_output):
        logits, = ctx.saved_tensors
        k = ctx.k
        eps= ctx.eps
        dtype = grad_output.dtype
        device = grad_output.device
        logits_np = logits.cpu().detach().numpy()
        grad_output_np = grad_output.cpu().detach().numpy()
        n1 = softtopk_np(logits_np + eps * grad_output_np, k)
        n2 = softtopk_np(logits_np - eps * grad_output_np, k)
        grad_np = (n1 - n2) / (2 * eps)
        grad = torch.from_numpy(grad_np).type(dtype).to(device)
        return grad, None, None

def sample_topk_generic(logits, k, add_gumbel_noise=False, tau=1.0, eps=1e-2):
    uniforms = torch.empty_like(logits).float().uniform_().clamp_(EPS, 1 - EPS)
    if add_gumbel_noise:
        gumbels = -torch.log(-torch.log(uniforms))
        noisy_logits = logits + gumbels
    else:
        noisy_logits = logits
    soft_sample = SoftTopK.apply(noisy_logits / tau, k, eps)
    _, topk_indices = torch.topk(noisy_logits, k, dim=-1)
    hard_sample = torch.zeros_like(logits).scatter_(-1, topk_indices, 1.0)
    hard_sample = (hard_sample - soft_sample).detach() + soft_sample
    return hard_sample, soft_sample

class RelaxedTopK(torch.nn.Module):
    def __init__(self, tau=1.0, eps=1e-6):
        super(RelaxedTopK, self).__init__()
        self.tau = tau
        self.eps = eps

    def forward(self, scores, k, add_gumbel_noise=False):
        if add_gumbel_noise:
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-9) + 1e-9)
        else:
            gumbel_noise = 0.0
        scores = scores + gumbel_noise * self.tau

        # continuous top k
        khot = torch.zeros_like(scores)
        onehot_approx = torch.zeros_like(scores)
        for i in range(k):
            khot_mask = torch.max(1.0 - onehot_approx, torch.tensor([1e-9], device=scores.device))
            scores = scores + torch.log(khot_mask)
            onehot_approx = torch.nn.functional.softmax(scores, dim=1)
            khot = khot + onehot_approx

        # straight through
        khot_hard = torch.zeros_like(khot)
        val, ind = torch.topk(khot, k, dim=1)
        khot_hard = khot_hard.scatter_(1, ind, 1)
        binary_mask = khot_hard - khot.detach() + khot
        soft_probs = khot

        return binary_mask, soft_probs

class SwiGLU(nn.Module):
    def __init__(self, dim, expand_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, expand_dim, bias=False)
        self.w3 = nn.Linear(dim, expand_dim, bias=False)
    def forward(self, x):
        return F.silu(self.w1(x)) * self.w3(x)


# ===================================================================
# 3. Main TokenPooler and TokenUnpooler Modules (Mask-based)
# ===================================================================
class TokenPooler(nn.Module):
    def __init__(self, dim: int, n_head: int = 8, ffn_mult: int = 1, dropout: float = 0.1, k=None, r=None, temperature=1.0, **attn_kwargs):
        super().__init__()
        self.dim = dim
        self.norm1 = RMSNorm(dim)
        self.attn = SelfAttention(dim, n_head=n_head, dropout=dropout, **attn_kwargs)
        
        self.norm2 = RMSNorm(dim)
        expand_dim = dim * ffn_mult
        self.swiglu = SwiGLU(dim, expand_dim)
        self.scorer = nn.Linear(expand_dim, 1, bias=False)
        self.k = k
        self.r = r
        self.temperature = temperature
        self.relaxed_topk = RelaxedTopK(tau=temperature, eps=1e-6)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [B, T, C].
            k, r: Define the number of tokens to keep.
            temperature: Temperature for Gumbel-Top-k.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - pooled_tokens (torch.Tensor): A sparse representation where non-selected tokens are zeroed out. Shape: [B, T, C].
                - binary_mask (torch.Tensor): The binary mask used for selection. Shape: [B, T].
        """
        B, T, C = x.shape
        assert C == self.dim
        assert self.k is not None or self.r is not None, "Either k or r must be provided."
        
        if self.r is not None:
            k = max(1, int(T * self.r))
        else:
            k = self.k

        # Scoring pipeline
        residual = x
        x_norm = self.norm1(x)
        attn_out = self.attn(x_norm)
        x = residual + attn_out

        x_norm = self.norm2(x)
        x_expanded = self.swiglu(x_norm)
        scores = self.scorer(x_expanded).squeeze(-1) # -> [B, T]

        # Get binary mask using STE
        # binary_mask = GumbelTopK_STE.apply(scores, k, self.temperature) # -> [B, T]
        binary_mask, soft_probs = sample_topk_generic(scores, k, add_gumbel_noise=self.training, tau=self.temperature)
        # binary_mask, soft_probs = self.relaxed_topk(scores, k, add_gumbel_noise=self.training)
        
        # Apply mask to get a "sparse" tensor.
        # The actual compression happens when the decoder learns to ignore masked parts.
        pooled_tokens = x[binary_mask.bool()]#residual # * binary_mask.unsqueeze(-1)
        pooled_tokens = pooled_tokens.view(B, k, C)
        
        return residual, pooled_tokens, binary_mask, soft_probs

class TokenUnpooler(nn.Module):
    def __init__(self, dim: int, use_soft_probs: bool = False):
        super().__init__()
        self.dim = dim
        # if not use_soft_probs:
        self.mask_token = nn.Parameter(torch.randn(dim))
        self.use_soft_probs = use_soft_probs

    def forward(self, residual: torch.Tensor, pooled_tokens: torch.Tensor, binary_mask: torch.Tensor, soft_probs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pooled_tokens (torch.Tensor): The sparse tensor from TokenPooler. Shape: [B, T, C].
            binary_mask (torch.Tensor): The binary selection mask. Shape: [B, T].

        Returns:
            torch.Tensor: The unpooled sequence where masked positions are filled. Shape: [B, T, C].
        """
        B, T, C_ = residual.shape
        B, k, C = pooled_tokens.shape
        assert C_ == self.dim
        assert C == self.dim

        # # Invert the mask to find positions that need to be filled
        # if self.use_soft_probs:
        #     mask = soft_probs.unsqueeze(-1)
        #     if self.training:
        #         std = pooled_tokens.std(dim=(-1, -2), keepdim=True)
        #     else:
        #         std = pooled_tokens[binary_mask.bool()].std(dim=(-1, -2), keepdim=True)
        #     mask_fill = torch.randn_like(pooled_tokens) * std
        # else:
        #     mask = binary_mask.unsqueeze(-1)
        #     # Create a tensor of mask tokens
        #     mask_fill = self.mask_token.view(1, 1, C).expand(B, T, -1)
        # Invert the mask to find positions that need to be filled

        mask = binary_mask.unsqueeze(-1)
        # Create a tensor of mask tokens
        mask_fill = self.mask_token.view(1, 1, C).expand(B, T, -1)
        inverse_mask = 1.0 - mask # -> [B, T]        
        
        # if self.training:
        #     unpooled_tokens = residual.to(pooled_tokens.dtype)
        # else:
        #     unpooled_tokens = torch.zeros_like(residual).to(pooled_tokens.dtype)
        # unpooled_tokens[binary_mask.bool()] = pooled_tokens.view(-1, C)
        # unpooled_tokens = unpooled_tokens.view(B, T, C)

        # Fill the zeroed-out positions with the mask token
        # This combines the selected tokens from pooled_tokens and the mask_token
        unpooled_tensor = residual * mask + mask_fill * inverse_mask
        
        return unpooled_tensor