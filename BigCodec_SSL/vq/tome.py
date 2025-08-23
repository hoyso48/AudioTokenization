import torch
import torch.nn as nn
import math
from typing import Tuple, Callable, List, Any
import torch.nn.functional as F

class OurToMe(nn.Module):
    """
    PyTorch nn.Module처럼 동작하도록 설계된 ToMe 레이어.
    - 인접 토큰만 병합
    - 최대 병합 사이즈 'm' 제한
    - 'iterations'에 걸쳐 점진적 병합
    - Merge & Unmerge 지원
    """

    def __init__(self, r: float, m: int, iterations: int):
        """
        알고리즘을 위한 하이퍼파라미터를 초기화합니다.
        
        Args:
            r (float): 줄일 토큰의 비율 (0.0 ~ 1.0).
            m (int): 하나의 토큰이 가질 수 있는 최대 원본 토큰 수.
            iterations (int): 병합을 수행할 반복 횟수.
        """
        super().__init__()
        if not (0.0 <= r <= 1.0):
            raise ValueError("r must be between 0.0 and 1.0")
        if m < 1:
            raise ValueError("m must be at least 1")
        if iterations < 1:
            raise ValueError("iterations must be at least 1")
        if m == 1 and r > 0:
            raise ValueError("Cannot merge tokens if m=1 and r > 0")
        
        self.r = r
        self.m = m
        self.iterations = iterations

    def _validate_runtime_args(self, N: int):
        """런타임에 결정되는 인자에 대한 유효성을 검사합니다."""
        if self.m > 1:
            r_max = 1 - 1 / (2**math.floor(math.log2(self.m)))
            if self.r > r_max:
                raise ValueError(f"r={self.r} is too high for m={self.m}. Max r for this m is {r_max:.4f}")

        num_total_tokens_to_reduce = int(self.r * N)
        if num_total_tokens_to_reduce % 2 != 0:
            raise ValueError(f"r * N ({self.r} * {N} = {self.r*N}) must result in an even number of tokens to reduce.")

    @staticmethod
    @torch.no_grad()
    def _get_merge_indices(
        metric: torch.Tensor, size: torch.Tensor, m: int, num_pairs_to_merge: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        병합할 인접 쌍의 인덱스를 계산합니다. (no_grad 컨텍스트)
        짝수 인덱스와 홀수 인덱스 쌍의 유사도를 계산하여 Top-K를 선택합니다.
        """
        B, N, C = metric.shape
        
        # 짝수/홀수 토큰 및 사이즈 분리
        # 토큰 수가 홀수일 경우 마지막 토큰은 고려하지 않음
        L = N // 2
        even_indices = torch.arange(0, 2 * L, 2, device=metric.device)
        odd_indices = torch.arange(1, 2 * L, 2, device=metric.device)

        even_tokens = metric[:, even_indices]
        odd_tokens = metric[:, odd_indices]
        
        even_sizes = size[:, even_indices]
        odd_sizes = size[:, odd_indices]

        # 1. 인접 유사도 계산 (코사인 유사도 사용)
        sim = torch.nn.functional.cosine_similarity(even_tokens, odd_tokens, dim=-1) # (B, L)

        # 2. 'm' 제약 조건에 따른 마스킹
        future_size = even_sizes + odd_sizes
        mask = (future_size > m).squeeze(-1)
        sim.masked_fill_(mask, -float('inf'))

        # 3. 병합할 쌍이 부족할 경우 처리
        num_possible_pairs = (~mask).sum(dim=1)
        k = min(num_pairs_to_merge, num_possible_pairs.min().item())
        if k < num_pairs_to_merge:
             print(f"Warning: Not enough pairs to merge. Merging {k} pairs instead of {num_pairs_to_merge}.")
        if k == 0:
            return torch.tensor([], dtype=torch.long, device=metric.device), \
                   torch.tensor([], dtype=torch.long, device=metric.device)

        # 4. Top-K 선택
        _, topk_indices = torch.topk(sim, k=k, dim=1) # (B, k)

        # topk_indices는 (0, L-1) 범위의 인덱스. 원래 짝수/홀수 인덱스로 변환 필요.
        merge_even_indices = torch.gather(even_indices.expand(B, -1), 1, topk_indices)
        merge_odd_indices = torch.gather(odd_indices.expand(B, -1), 1, topk_indices)
        
        return merge_even_indices, merge_odd_indices


    @staticmethod
    def _merge_step(
        x: torch.Tensor, size: torch.Tensor, merge_even_indices: torch.Tensor, merge_odd_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        실제 텐서 병합을 수행합니다. (그래디언트 추적)
        """
        B, N, C = x.shape
        
        # 병합될 토큰을 표시하는 마스크 생성
        merged_mask = torch.zeros(B, N, dtype=torch.bool, device=x.device)
        merged_mask.scatter_(1, merge_even_indices, True)
        merged_mask.scatter_(1, merge_odd_indices, True)
        
        # 병합되지 않을 토큰들
        unmerged_indices = (~merged_mask).nonzero(as_tuple=True)[1].view(B, -1)
        unmerged_tokens = torch.gather(x, 1, unmerged_indices.unsqueeze(-1).expand(-1, -1, C))
        unmerged_sizes = torch.gather(size, 1, unmerged_indices.unsqueeze(-1))

        # 병합될 토큰들 (짝수 쪽으로 병합)
        even_toks = torch.gather(x, 1, merge_even_indices.unsqueeze(-1).expand(-1, -1, C))
        odd_toks = torch.gather(x, 1, merge_odd_indices.unsqueeze(-1).expand(-1, -1, C))
        
        even_s = torch.gather(size, 1, merge_even_indices.unsqueeze(-1))
        odd_s = torch.gather(size, 1, merge_odd_indices.unsqueeze(-1))

        # 가중 평균으로 병합
        new_size = even_s + odd_s
        merged_toks = (even_toks * even_s + odd_toks * odd_s) / new_size

        # 새로운 텐서 생성
        # 병합된 토큰은 원래 짝수 인덱스 위치에 들어가고, unmerged 토큰과 합쳐짐
        # 순서를 유지하기 위해, 모든 토큰을 모은 뒤 정렬
        
        new_x = torch.cat([unmerged_tokens, merged_toks], dim=1)
        new_size = torch.cat([unmerged_sizes, new_size], dim=1)
        
        # 다음 단계와 unmerge를 위한 정보
        # 병합 후 토큰들의 원래 인덱스를 추적
        original_indices = torch.cat([
            unmerged_indices,
            merge_even_indices # 병합된 토큰은 짝수 인덱스를 대표
        ], dim=1)
        
        # 정렬하여 순서 유지
        perm = original_indices.argsort(dim=1)
        new_x = torch.gather(new_x, 1, perm.unsqueeze(-1).expand(-1, -1, C))
        new_size = torch.gather(new_size, 1, perm.unsqueeze(-1))
        
        # Unmerge를 위한 정보 저장
        merge_info = {
            "original_N": N,
            "merge_even_indices": merge_even_indices,
            "merge_odd_indices": merge_odd_indices,
        }
        # print(merge_info)
        return new_x, new_size, merge_info

    @staticmethod
    def _unmerge_step(y: torch.Tensor, merge_info: dict) -> torch.Tensor:
        """
        한 단계의 병합을 되돌립니다.
        """
        B, N_merged, C = y.shape
        original_N = merge_info["original_N"]
        merge_even_indices = merge_info["merge_even_indices"]
        merge_odd_indices = merge_info["merge_odd_indices"]

        # 병합되었던 토큰들을 찾기
        # 현재 y에서 병합된 토큰들이 어느 위치에 있는지 알아내야 함
        
        # unmerge를 위한 마스크 생성
        merged_mask_original = torch.zeros(B, original_N, dtype=torch.bool, device=y.device)
        merged_mask_original.scatter_(1, merge_even_indices, True)
        merged_mask_original.scatter_(1, merge_odd_indices, True)
        
        unmerged_indices_original = (~merged_mask_original).nonzero(as_tuple=True)[1].view(B, -1)
        
        # 병합 후 시퀀스에서 unmerged 토큰과 merged 토큰의 위치를 알아냄
        temp_indices = torch.cat([
            unmerged_indices_original,
            merge_even_indices
        ], dim=1)
        perm = temp_indices.argsort(dim=1)
        inv_perm = perm.argsort(dim=1) # 역순열

        # y를 원래 순서(unmerged, merged)로 되돌림
        y_resorted = torch.gather(y, 1, inv_perm.unsqueeze(-1).expand(-1, -1, C))
        
        num_unmerged = unmerged_indices_original.shape[1]
        unmerged_toks = y_resorted[:, :num_unmerged]
        merged_toks = y_resorted[:, num_unmerged:]
        
        # 원래 크기의 텐서 생성
        reconstructed_y = torch.zeros(B, original_N, C, device=y.device, dtype=y.dtype)
        
        # unmerged 토큰 복원
        reconstructed_y.scatter_(1, unmerged_indices_original.unsqueeze(-1).expand(-1, -1, C), unmerged_toks)
        
        # merged 토큰 복원 (짝수, 홀수 위치 모두에 동일 값 복사)
        reconstructed_y.scatter_(1, merge_even_indices.unsqueeze(-1).expand(-1, -1, C), merged_toks)
        reconstructed_y.scatter_(1, merge_odd_indices.unsqueeze(-1).expand(-1, -1, C), merged_toks)
        
        return reconstructed_y

    def merge(self, metric: torch.Tensor) -> Tuple[torch.Tensor, Callable, torch.Tensor]:
        """
        입력 텐서에 대해 병합을 수행합니다.
        
        Args:
            metric (torch.Tensor): 병합할 토큰 텐서 (B, N, C).
            
        Returns:
            - merged_tensor (torch.Tensor): 최종 병합된 텐서.
            - unmerge_fn (Callable): 병합을 되돌리는 함수.
            - final_size (torch.Tensor): 병합된 각 토큰의 최종 size.
        """
        B, N, C = metric.shape
        device = metric.device

        # 런타임 유효성 검사
        self._validate_runtime_args(N)

        num_total_pairs_to_merge = int(self.r * N)

        if num_total_pairs_to_merge == 0:
            size = torch.ones(B, N, 1, device=device)
            return metric, lambda x: x, size

        # 각 이터레이션에 병합할 쌍의 수 분배
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
        """nn.Module의 forward와 동일한 역할을 합니다."""
        return self.merge(metric)


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