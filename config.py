"""VSR 视频字幕/水印移除引擎配置。

从环境变量 + 命令行参数加载，提供引擎运行所需的全部配置。

v0.1 设计（VSR_VIDEO_SUBTITLE_REMOVER_INTEGRATION_ANALYSIS.md §4）：
    - engine_id: video_subtitle_remover_d1e2f3a4（目录/manifest/config 三同名契约）
    - 端口: 7012（7001-7008 已占用，模板 7010/7011 预留）
    - 模型布局: video/vsr/（4 算法子目录 + V5/ch_det OCR 目录）

双源真相消除策略（同 LTX-2.5 引擎）：
    - MODEL_SUBDIR 默认读环境变量 AIRELIABLE_MODEL_LAYOUT_BASE
      （由调度层 EngineManager 启动子进程时同步 manifest.models.model_layout.base）
    - 优先级：环境变量 > 硬编码 fallback
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path

from aireliable_engine import EngineConfig as BaseEngineConfig


# ===== 常量 =====

ENGINE_ID = "video_subtitle_remover_d1e2f3a4"
ENGINE_NAME = "VSR-VideoSubtitleRemover"
CAPABILITY = "video-editing"
ENGINE_TYPE = "video"

# 默认端口（与 config/engines/video_subtitle_remover_d1e2f3a4.yaml 一致）
DEFAULT_PORT = 7012

# 模型目录布局（相对 MODELS_ROOT）
# 优先读环境变量 AIRELIABLE_MODEL_LAYOUT_BASE（平台注入）
_DEFAULT_MODEL_SUBDIR = "video/vsr/"


def _resolve_model_subdir() -> str:
    """解析模型子目录（优先读环境变量）。"""
    env_val = os.environ.get("AIRELIABLE_MODEL_LAYOUT_BASE")
    if env_val:
        return env_val.rstrip("/") + "/"
    return _DEFAULT_MODEL_SUBDIR


# 运行时常量
MODEL_SUBDIR = _resolve_model_subdir()


def _detect_gpu_vram_gb() -> float:
    """检测当前 GPU 显存（GB），无 GPU 或检测失败返回 0.0。"""
    try:
        import torch
        if torch.cuda.is_available():
            total_bytes = torch.cuda.get_device_properties(0).total_memory
            return total_bytes / (1024 ** 3)
    except Exception:
        pass
    return 0.0


@dataclass
class EngineConfig(BaseEngineConfig):
    """VSR 视频字幕移除引擎运行配置。"""

    # ===== 基础（身份 / 端口默认值由本引擎覆盖，其余公共字段见基类）=====
    engine_id: str = ENGINE_ID
    engine_name: str = ENGINE_NAME
    capability: str = CAPABILITY
    engine_type: str = ENGINE_TYPE
    port: int = DEFAULT_PORT

    # ===== 推导路径 =====
    model_dir: Path = field(init=False)
    log_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        """推导子路径。"""
        # 模型目录（video/vsr/，含 4 算法子目录 + V5/ch_det OCR 目录）
        self.model_dir = self.models_root / MODEL_SUBDIR

        # 日志目录（v0.2.15+ 统一到 logs/engines/）
        self.log_dir = self.project_root / "logs" / "engines"

        # 确保运行时目录存在
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def resolve_device(self) -> str:
        """解析推理设备（与 upstream HardwareAccelerator 语义对齐）。

        - auto：cuda 可用则 cuda:0，否则 cpu
        - cpu：强制 cpu
        - cuda:0 / cuda:1：显式指定

        Returns:
            解析后的设备字符串（"cuda:0" / "cpu"）
        """
        device = self.device.strip()
        if device.startswith("cuda") or device == "cpu":
            return device
        # auto
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda:0"
        except Exception:
            pass
        return "cpu"

    def to_dict(self) -> dict:
        """转 dict（用于日志/调试）。"""
        return {
            "engine_id": self.engine_id,
            "engine_name": self.engine_name,
            "capability": self.capability,
            "engine_type": self.engine_type,
            "port": self.port,
            "host": self.host,
            "models_root": str(self.models_root),
            "model_dir": str(self.model_dir),
            "output_root": str(self.output_root),
            "log_dir": str(self.log_dir),
            "device": self.device,
            "log_level": self.log_level,
        }


def parse_args() -> EngineConfig:
    """解析命令行参数 + 环境变量，构造 EngineConfig。"""
    parser = argparse.ArgumentParser(description="VSR 视频字幕/水印移除引擎")

    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"HTTP 端口（默认 {DEFAULT_PORT}）")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="绑定地址（默认 127.0.0.1）")
    parser.add_argument("--device", type=str, default="auto",
                        help="推理设备：auto / cpu / cuda:0（默认 auto）")
    parser.add_argument("--log-level", type=str, default="INFO",
                        help="日志级别（默认 INFO）")
    parser.add_argument("--models-root", type=str, default=None,
                        help="模型根目录（默认 models/）")

    args = parser.parse_args()

    models_root = Path(args.models_root or os.environ.get("AIRELIABLE_MODELS_ROOT", "models"))

    return EngineConfig(
        port=args.port,
        host=args.host,
        device=args.device,
        log_level=args.log_level,
        models_root=models_root,
    )
