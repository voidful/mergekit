# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

from typing import List, Optional

import torch

from mergekit.merge_methods.easy_define import merge_method
from mergekit.merge_methods.generalized_task_arithmetic import (
    get_mask as sign_consensus_mask,
)

@merge_method(
    name="sce",
    pretty_name="SCE",
    reference_url="https://arxiv.org/abs/2408.07990",
)
def sce_merge(
    tensors: List[torch.Tensor],
    base_tensor: torch.Tensor,
    int8_mask: bool = False,
    select_topk: float = 1.0,  # 當 select_topk < 1 時啟用動態選取門檻
) -> torch.Tensor:
    if not tensors:
        return base_tensor
    mask_dtype = torch.int8 if int8_mask else base_tensor.dtype
    # 計算每個來源模型相對於 pivot 模型的更新向量 (task vectors)
    task_vectors = torch.stack([t - base_tensor for t in tensors], dim=0)

    # --- 動態選取門檻 (Adaptive Thresholding) ---
    # 根據各參數位置更新向量的變異數，自適應篩選出顯著更新位置
    if select_topk < 1:
        mask = sce_mask(task_vectors, select_topk, mask_dtype)
        task_vectors = task_vectors * mask.unsqueeze(0)

    # --- Erase 步驟：消除不同模型更新方向不一致的部分 ---
    erase_mask = sign_consensus_mask(task_vectors, method="sum", mask_dtype=mask_dtype)

    # --- 跨層正則化與自適應學習率 ---
    # 計算每個來源模型在各參數位置上的融合權重
    tv_weights = sce_weight(task_vectors, reg_factor=0.1)
    while tv_weights.dim() < task_vectors.dim():
        tv_weights = tv_weights.unsqueeze(-1)

    # 利用消除掩碼避免少數極端更新造成干擾
    erased_weights = tv_weights * erase_mask
    merged_tv = (task_vectors * erased_weights).sum(dim=0)
    final_tv = merged_tv / torch.sum(erased_weights, dim=0).clamp(min=1e-6)

    return base_tensor + final_tv

def sce_weight(tvs: torch.Tensor, reg_factor: float = 0.1) -> torch.Tensor:
    """
    透過計算均方值作為基礎權重，並加入兩項改進：
    1. 跨層正則化：混合全局平均值以平滑極端局部差異。
    2. 自適應學習率：根據局部變異數調整每個參數的權重，降低高變異區域的影響。
    """
    # 基礎權重：均方值
    weights = torch.mean(tvs**2, dim=list(range(1, tvs.dim())))
    # 全局平均權重
    avg_weight = torch.mean(weights)
    # 跨層正則化：局部權重與全局平均混合 (reg_factor 控制混合程度)
    weights = (1 - reg_factor) * weights + reg_factor * avg_weight

    # 自適應學習率：根據每個位置的變異數調整權重
    var = torch.var(tvs, dim=list(range(1, tvs.dim())), unbiased=False)
    adaptive_factor = 1.0 / (1.0 + var)  # 當變異數大時，adaptive_factor 趨近於較小值
    weights = weights * adaptive_factor

    weight_sum = torch.sum(weights).item()
    if abs(weight_sum) < 1e-6:
        return torch.ones_like(weights) / weights.shape[0]
    return weights / weight_sum

def sce_mask(
    tvs: torch.Tensor, density: float, mask_dtype: Optional[torch.dtype] = None
):
    """
    根據動態選取門檻篩選參數：
    1. 計算所有來源模型對應參數位置的變異數。
    2. 利用分位數 (1 - density) 作為門檻，只有變異數大於此門檻的位置視為顯著更新。
    """
    if density <= 0:
        return torch.zeros_like(tvs, dtype=mask_dtype)
    if density >= 1:
        return torch.ones_like(tvs, dtype=mask_dtype)

    # 計算各參數位置的變異數 (跨所有模型)
    var = torch.var(tvs, dim=0, unbiased=False)
    # 動態門檻：取 1 - density 分位數
    threshold = torch.quantile(var.flatten(), 1 - density)
    mask = (var >= threshold).to(dtype=mask_dtype)
    return mask

def iterative_sce_merge(
    tensors: List[torch.Tensor],
    base_tensor: torch.Tensor,
    int8_mask: bool = False,
    select_topk: float = 1.0,
    num_iterations: int = 3,
    momentum: float = 0.9,
) -> torch.Tensor:
    """
    利用多次迭代融合與動量機制平滑更新結果：
    - 每次根據當前模型進行 SCE 融合，計算更新向量。
    - 利用動量平滑當前更新與上次更新的累積結果，
      進而減少單次融合中可能出現的劇烈波動。
    - 迭代完成後返回最終融合模型參數。
    """
    current_tensor = base_tensor.clone()
    aggregated_update = torch.zeros_like(base_tensor)
    for _ in range(num_iterations):
        # 使用當前模型作為 pivot，計算一次 SCE 融合更新
        merged_tensor = sce_merge(tensors, current_tensor, int8_mask, select_topk)
        update = merged_tensor - current_tensor
        # 動量平滑更新：累積前後更新
        aggregated_update = momentum * aggregated_update + (1 - momentum) * update
        current_tensor = current_tensor + aggregated_update
    return current_tensor


# 動態選取門檻（Adaptive Thresholding）：
# 修改了 sce_mask 函數，計算各參數位置的變異數，並以變異數的 (1 - density) 分位數作為門檻。
# 只有當變異數超過該門檻時，該位置才被認為是顯著的，從而有效過濾掉噪音資訊。
# 跨層正則化與自適應學習率（Cross-Layer Regularization & Adaptive Learning Rate）：
# 在 sce_weight 函數中，先計算基礎權重（均方值），再混合全局平均值以平滑局部極端差異（跨層正則化）。
# 接著根據每個位置的變異數計算自適應因子（adaptive_factor），調整各位置權重，相當於根據局部統計資訊自適應調整“學習率”。
# 引入動量機制與多次迭代融合（Momentum & Iterative Merging）：
# 新增 iterative_sce_merge 函數，將融合過程設計為多次迭代，每次利用 SCE 融合計算更新。
# 利用動量公式平滑更新，累積歷史更新資訊，降低單次融合的劇烈波動，進一步提高融合結果的穩定性。
# 整體設計思路：
# 將原有 SCE 融合方法中各步驟（選取、計算、消除）與進一步優化策略（動態門檻、正則化、自適應調整、動量迭代）結合，達到更平滑、穩健的融合效果。
# 各改進策略分別針對參數噪音、局部極端差異以及層間不一致性進行補償，最終希望能更好地融合各來源模型的優點，降低模型退化風險。
