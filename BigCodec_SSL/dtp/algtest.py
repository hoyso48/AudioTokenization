from typing import List, Tuple
import heapq
import math
import random

def validate_non_adjacent(indices: List[int]) -> bool:
    """Return True if no two indices are adjacent."""
    s = set(indices)
    return all((i - 1) not in s and (i + 1) not in s for i in s)

def sum_scores(scores: List[float], indices: List[int]) -> float:
    return float(sum(scores[i] for i in indices))

def select_pairs_dp(scores: List[float], k: int) -> List[int]:
    """
    Exact optimal: pick exactly k non-adjacent indices maximizing sum.
    Tie-breaking: prefer 'take' on ties (leftmost bias).
    Time: O(m*k), Space: O(m*k) for backpointers (safe for given scale).
    """
    m = len(scores)
    if k == 0:
        return []
    if k > (m + 1) // 2:
        raise ValueError("k is infeasible for given length with non-adjacent constraint.")

    NEG_INF = float("-inf")
    # take[i][t] = True if best at (i, t) takes i
    take = [[False] * (k + 1) for _ in range(m)]

    # rolling dp for values
    dp_prevprev = [NEG_INF] * (k + 1)  # dp[i-2][*]
    dp_prevprev[0] = 0.0
    dp_prev = [NEG_INF] * (k + 1)      # dp[i-1][*]
    dp_prev[0] = 0.0

    for i in range(m):
        dp_curr = dp_prev[:]  # skipping i by default
        for t in range(1, k + 1):
            take_val = (dp_prevprev[t - 1] + scores[i]) if dp_prevprev[t - 1] != NEG_INF else NEG_INF
            # Tie-breaking: prefer take on ties for leftmost bias
            if take_val >= dp_curr[t]:
                dp_curr[t] = take_val
                take[i][t] = True
        dp_prevprev, dp_prev = dp_prev, dp_curr

    if dp_prev[k] == NEG_INF:
        raise RuntimeError("DP failed to find a feasible solution though k should be feasible.")

    # Backtrack
    res: List[int] = []
    i, t = m - 1, k
    while i >= 0 and t > 0:
        if take[i][t]:
            res.append(i)
            i -= 2
            t -= 1
        else:
            i -= 1
    res.reverse()
    assert len(res) == k and validate_non_adjacent(res)
    return res

def select_pairs_greedy(scores: List[float], k: int) -> List[int]:
    """
    Greedy: repeatedly pick global max, mask neighbors.
    Tie-breaking: leftmost via heap key (-score, index).
    Time: O(m log m).
    """
    m = len(scores)
    if k == 0:
        return []
    if k > (m + 1) // 2:
        raise ValueError("k is infeasible for given length with non-adjacent constraint.")

    heap: List[Tuple[float, int]] = [(-scores[i], i) for i in range(m)]
    heapq.heapify(heap)
    blocked = [False] * m
    chosen: List[int] = []

    while len(chosen) < k:
        if not heap:
            raise RuntimeError("Heap exhausted before reaching k; unexpected under given assumptions.")
        neg_s, i = heapq.heappop(heap)
        if blocked[i]:
            continue
        # choose i
        chosen.append(i)
        # mask i and neighbors
        blocked[i] = True
        if i - 1 >= 0:
            blocked[i - 1] = True
        if i + 1 < m:
            blocked[i + 1] = True

    chosen.sort()
    assert len(chosen) == k and validate_non_adjacent(chosen)
    return chosen

def select_pairs_bipartite(scores: List[float], k: int) -> List[int]:
    """
    Parity split: try even indices and odd indices separately, take better sum.
    Time: O(m log m) due to sorting (can be O(m) with selection).
    """
    m = len(scores)
    if k == 0:
        return []
    if k > (m + 1) // 2:
        raise ValueError("k is infeasible for given length with non-adjacent constraint.")

    even_idxs = [i for i in range(0, m, 2)]
    odd_idxs = [i for i in range(1, m, 2)]

    def topk_from(indices: List[int]) -> List[int]:
        if len(indices) < k:
            return []
        top = sorted(indices, key=lambda i: (-scores[i], i))[:k]
        top.sort()
        return top

    cand_even = topk_from(even_idxs)
    cand_odd = topk_from(odd_idxs)

    if not cand_even and not cand_odd:
        raise RuntimeError("Both parities have insufficient capacity for k; unexpected under given assumptions.")

    if cand_even and not cand_odd:
        return cand_even
    if cand_odd and not cand_even:
        return cand_odd

    sum_even = sum_scores(scores, cand_even)
    sum_odd = sum_scores(scores, cand_odd)
    return cand_even if sum_even >= sum_odd else cand_odd

def maxpool_nms_indices(scores: List[float]) -> List[int]:
    """
    1D NMS with window=3, leftmost tie:
    keep i if scores[i] > scores[i-1] and scores[i] >= scores[i+1]
    with out-of-range neighbors treated as -inf.
    """
    m = len(scores)
    kept: List[int] = []
    for i in range(m):
        left = scores[i - 1] if i - 1 >= 0 else float("-inf")
        right = scores[i + 1] if i + 1 < m else float("-inf")
        if scores[i] > left and scores[i] >= right:
            kept.append(i)
    # kept are non-adjacent by construction
    return kept

def select_pairs_maxpool_with_fallback(scores: List[float], k: int, fallback: str = "greedy") -> List[int]:
    """
    MaxPool prefilter (leftmost-tie) then top-k among survivors.
    If survivors < k, fallback to fill deficit while respecting chosen set.
    fallback in {"greedy", "dp"}.
    """
    m = len(scores)
    if k == 0:
        return []
    if k > (m + 1) // 2:
        raise ValueError("k is infeasible for given length with non-adjacent constraint.")

    survivors = maxpool_nms_indices(scores)
    if len(survivors) >= k:
        # top-k among survivors
        picked = sorted(survivors, key=lambda i: (-scores[i], i))[:k]
        picked.sort()
        assert validate_non_adjacent(picked)
        return picked

    # Need fallback to reach k
    picked = survivors[:]  # already non-adjacent
    blocked = [False] * m
    for i in picked:
        blocked[i] = True
        if i - 1 >= 0:
            blocked[i - 1] = True
        if i + 1 < m:
            blocked[i + 1] = True

    remaining_indices = [i for i in range(m) if not blocked[i]]
    if not remaining_indices and len(picked) < k:
        # Extremely adversarial, but under assumptions this should not happen
        raise RuntimeError("No remaining indices to fill after NMS; unexpected under given assumptions.")

    if fallback == "greedy":
        # Greedy on the remaining, respecting blocked mask
        heap: List[Tuple[float, int]] = [(-scores[i], i) for i in remaining_indices]
        heapq.heapify(heap)
        while len(picked) < k:
            if not heap:
                raise RuntimeError("Greedy fallback exhausted prematurely.")
            neg_s, i = heapq.heappop(heap)
            if blocked[i]:
                continue
            picked.append(i)
            blocked[i] = True
            if i - 1 >= 0:
                blocked[i - 1] = True
            if i + 1 < m:
                blocked[i + 1] = True
        picked.sort()
        assert validate_non_adjacent(picked)
        return picked

    elif fallback == "dp":
        # Run DP on the reduced path (mask blocked), selecting exactly (k - len(picked)) more
        # Build a contracted list of available positions with their scores
        avail = [i for i in range(m) if not blocked[i]]
        if not avail and len(picked) < k:
            raise RuntimeError("DP fallback has no available indices.")
        # Map contiguous blocks to subproblems because adjacency is only within blocks
        result_extra: List[int] = []
        need = k - len(picked)
        # Partition avail into contiguous blocks
        blocks: List[List[int]] = []
        curr: List[int] = []
        for idx in avail:
            if not curr or idx == curr[-1] + 1:
                curr.append(idx)
            else:
                blocks.append(curr)
                curr = [idx]
        if curr:
            blocks.append(curr)

        # Greedy block allocation by potential capacity, then DP per block until need is filled.
        for block in blocks:
            if need == 0:
                break
            b_scores = [scores[i] for i in block]
            cap = (len(block) + 1) // 2
            take_here = min(cap, need)
            if take_here == 0:
                continue
            # DP within block to pick exactly take_here
            extra = select_pairs_dp(b_scores, take_here)
            result_extra.extend(block[i] for i in extra)
            need -= take_here

        if need != 0:
            raise RuntimeError("DP fallback did not fill to k; unexpected under given assumptions.")

        picked.extend(result_extra)
        picked.sort()
        assert validate_non_adjacent(picked)
        return picked

    else:
        raise ValueError("fallback must be 'greedy' or 'dp'")

def benchmark_methods(scores: List[float], k: int):
    """Return selections and sums for each method."""
    methods = {
        "dp": select_pairs_dp,
        "greedy": select_pairs_greedy,
        "bipartite": select_pairs_bipartite,
        "maxpool_greedy": lambda s, kk: select_pairs_maxpool_with_fallback(s, kk, "greedy"),
        "maxpool_dp":     lambda s, kk: select_pairs_maxpool_with_fallback(s, kk, "dp"),
    }
    out = {}
    for name, fn in methods.items():
        idxs = fn(scores, k)
        out[name] = {
            "indices": idxs,
            "sum": sum_scores(scores, idxs),
            "valid": validate_non_adjacent(idxs),
        }
    return out

if __name__ == "__main__":
    # Quick sanity test
    random.seed(0)
    m = 30
    scores = [random.random() for _ in range(m)]
    k = 8
    res = benchmark_methods(scores, k)
    for name, v in res.items():
        print(name, v["sum"], v["indices"], "valid=", v["valid"])