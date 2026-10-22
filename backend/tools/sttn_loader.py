"""STTN 权重加载器（兼容 safetensors 与 PyTorch checkpoint）。

权重存在两种存储格式（本模块统一归一化为 netG 状态字典）：
    1. netG 包装格式：上游 VSR 原始仓库的 infer_model.pth / sttn.pth
       （torch.load 的 dict，顶层含 netG 键）
    2. 直接状态字典格式：Dev-Jahn/sttn-auto|sttn-det 社区仓库的
       safetensors（112 键直接就是 InpaintGenerator 的 state_dict，
       无 netG 包装）

读取路径：
    - *.safetensors → safetensors.torch.load_file
    - *.pth / *.pt  → torch.load（兼容 PyTorch 2.6+ weights_only=False）

提取策略：
    - 存在 netG 键 → 返回 state_dict["netG"]
    - 无 netG 键但本身是合法模型状态字典（含 encoder/decoder 键）
      → 原样返回（Dev-Jahn safetensors 格式）
    - 其余情况 → 抛 RuntimeError 并给出下载指引
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)


def load_sttn_netg(model_path: str, device: str = "cpu") -> dict:
    """加载 STTN 权重并返回 netG 状态字典。

    Args:
        model_path: 模型文件路径（.safetensors 或 .pth/.pt）
        device: 加载目标设备（默认 cpu，后续由 .to(device) 转移）

    Returns:
        netG 的 state_dict（OrderedDict[str, Tensor]）

    Raises:
        RuntimeError: 文件不存在 / 解析失败 / 缺少 netG 键时抛出，
            信息中包含缺失文件路径与下载指引。
    """
    if not os.path.isfile(model_path):
        raise RuntimeError(
            f"STTN 模型文件不存在：{model_path}\n"
            f"请到引擎详情页 → 模型管理 → 点击「下载缺失文件」按钮，"
            f"选择镜像源后下载（HuggingFace 仓库：Dev-Jahn/sttn-auto 或 Dev-Jahn/sttn-det）。"
        )

    lower = model_path.lower()
    try:
        if lower.endswith(".safetensors"):
            from safetensors.torch import load_file

            state_dict = load_file(model_path, device=device)
            logger.info(
                "STTN 权重加载（safetensors）：%s，键数 %d，含 netG=%s",
                model_path,
                len(state_dict),
                "netG" in state_dict,
            )
        else:
            state_dict = torch.load(model_path, map_location=device, weights_only=False)
            logger.info(
                "STTN 权重加载（torch.load）：%s，顶层类型 %s",
                model_path,
                type(state_dict).__name__,
            )
    except Exception as e:
        raise RuntimeError(f"STTN 权重解析失败：{model_path}（{e}）") from e

    # 提取 netG：
    #   1. netG 包装格式（上游 VSR 的 infer_model.pth）→ state_dict["netG"]
    #   2. 直接状态字典格式（Dev-Jahn safetensors，无 netG 包装）
    #      → 校验是否包含 InpaintGenerator 的 encoder/decoder 键，是则原样返回
    if isinstance(state_dict, dict):
        if "netG" in state_dict:
            netg = state_dict["netG"]
        elif _looks_like_sttn_state_dict(state_dict):
            logger.info(
                "STTN 权重为直接状态字典格式（无 netG 包装），原样作为 netG 使用：%s",
                model_path,
            )
            netg = state_dict
        else:
            netg = None
    else:
        netg = None
    if netg is None:
        raise RuntimeError(
            f"STTN 权重缺少 netG 键：{model_path}（可用键："
            f"{list(state_dict.keys())[:10] if isinstance(state_dict, dict) else '非 dict'}）\n"
            f"请确认下载的是 Dev-Jahn/sttn-auto（STTN-AUTO）或 Dev-Jahn/sttn-det（STTN-DET）权重。"
        )
    return netg


def _looks_like_sttn_state_dict(state_dict: dict) -> bool:
    """判断状态字典是否为 InpaintGenerator 的直接 state_dict（无 netG 包装）。

    InpaintGenerator 的关键特征键：encoder.0.weight（编码器首层卷积）、
    decoder.0.conv.weight（decoder 首层 deconv 的内部卷积）。
    """
    keys = set(state_dict.keys())
    return "encoder.0.weight" in keys and "decoder.0.conv.weight" in keys
