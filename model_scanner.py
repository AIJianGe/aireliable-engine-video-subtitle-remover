"""VSR 模型文件扫描器（4 算法 variant 版）。

扫描 models/video/vsr/ 目录，识别可用的 VSR 模型变体。

4 variant 设计（VSR_VIDEO_SUBTITLE_REMOVER_INTEGRATION_ANALYSIS.md §4）：
    - VSR/STTN-AUTO   ：sttn-auto/model.safetensors（智能擦除，无需 OCR）
    - VSR/STTN-DET    ：sttn-det/model.safetensors（检测式擦除，需 OCR）
    - VSR/LAMA        ：big-lama/big-lama.pt（图像级 inpainting，需 OCR）
    - VSR/PROPAINTER  ：propainter/ProPainter.pth（视频级 inpainting，需 OCR）

v0.2.11+ 方案 B 改造：
    优先调用平台 API（`GET /engines/{id}/variants`）获取 variant 可用性，
    平台不可达时 fallback 到本地文件扫描（检查标志文件存在性）。

本地 fallback 检查策略（遵循 project-rules.md 第二部分 §5）：
    - 只检查标志文件存在性，不严格校验大小
    - 让 variant 在 UI 中可见（可触发下载），不阻断启动
    - 完整性校验由平台 API（key_files 机制）或 wrapper.load() 兜底

模型目录结构（与 manifest.yaml model_files.target_dir 一致）：
    models/video/vsr/
    ├── sttn-auto/            ← vsr-sttn-auto（model.safetensors）
    ├── sttn-det/             ← vsr-sttn-det（model.safetensors）
    ├── big-lama/             ← vsr-lama（big-lama.pt）
    ├── propainter/           ← vsr-propainter（ProPainter.pth + raft-things.pth + recurrent_flow_completion.pth）
    └── V5/
        ├── ch_det/           ← vsr-ocr-server（inference.yml + inference.pdiparams + inference.json）
        └── ch_det_fast/      ← vsr-ocr-mobile（inference.yml + inference.pdiparams + inference.json）
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .schemas import VSRVariantInfo

logger = logging.getLogger(__name__)


# ===== 已知模型变体定义（4 variant，name 必须与 manifest.yaml variants[].name 完全一致）=====
# subdir 为相对 model_dir（video/vsr/）的子目录。
# ocr_marker 为 OCR 组标志（server/mobile 任一存在即可，exclusive_group 语义）。
_KNOWN_VARIANTS: list[dict] = [
    # ============================================================
    # 1. STTN-AUTO（智能擦除，默认，无需 OCR）
    # ============================================================
    {
        "name": "VSR/STTN-AUTO",
        "alias": "STTN-AUTO",
        "base_model": "STTN",
        "precision": "fp32",
        "loader": "sttn",
        "vram_required": "4GB",
        "subdir": "sttn-auto",
        "inpaint_mode": "sttn-auto",
        "requires_ocr": False,
        "marker_files": ["model.safetensors"],
    },
    # ============================================================
    # 2. STTN-DET（检测式擦除，需 OCR）
    # ============================================================
    {
        "name": "VSR/STTN-DET",
        "alias": "STTN-DET",
        "base_model": "STTN",
        "precision": "fp32",
        "loader": "sttn",
        "vram_required": "4GB",
        "subdir": "sttn-det",
        "inpaint_mode": "sttn-det",
        "requires_ocr": True,
        "marker_files": ["model.safetensors"],
    },
    # ============================================================
    # 3. LAMA（图像级 inpainting，需 OCR）
    # ============================================================
    {
        "name": "VSR/LAMA",
        "alias": "LAMA",
        "base_model": "LAMA",
        "precision": "fp32",
        "loader": "lama",
        "vram_required": "2GB",
        "subdir": "big-lama",
        "inpaint_mode": "lama",
        "requires_ocr": True,
        "marker_files": ["big-lama.pt"],
    },
    # ============================================================
    # 4. PROPAINTER（视频级 inpainting，需 OCR）
    # ============================================================
    {
        "name": "VSR/PROPAINTER",
        "alias": "PROPAINTER",
        "base_model": "ProPainter",
        "precision": "fp32",
        "loader": "propainter",
        "vram_required": "25GB",
        "subdir": "propainter",
        "inpaint_mode": "propainter",
        "requires_ocr": True,
        "marker_files": ["ProPainter.pth"],
    },
]

# OCR 组标志文件（exclusive_group vsr-ocr：server/mobile 任一存在即可）
_OCR_MARKERS: list[tuple[str, str]] = [
    ("V5/ch_det", "inference.yml"),
    ("V5/ch_det_fast", "inference.yml"),
]


class ModelScanner:
    """模型目录扫描器（4 算法 variant 版）。

    扫描 model_dir/ 下的算法子目录 + OCR 目录，识别已下载的 VSR 变体。

    v0.2.11+ 方案 B：优先调用平台 API，fallback 到本地扫描。
    """

    def __init__(
        self,
        models_root: Path,
        model_dir: Path,
        engine=None,
        logger: Optional[logging.Logger] = None,
    ):
        """初始化扫描器。

        Args:
            models_root: 模型根目录（如 models/）
            model_dir: VSR 模型目录（video/vsr/）
            engine: 可选的 BaseEngine 实例（v0.2.11+ 方案 B 注入）。
                传入时优先调用平台 API 检测 variant；
                为 None 时走 fallback 本地扫描逻辑。
            logger: 可选的日志器
        """
        self.models_root = models_root
        self.model_dir = model_dir
        self.engine = engine
        self._logger = logger or logging.getLogger(__name__)

    async def scan_variants_async(self) -> list[VSRVariantInfo]:
        """v0.2.11+ 方案 B：异步扫描，优先调用平台 API。"""
        if self.engine is not None:
            platform_variants = await self.engine.get_variants()
            if platform_variants is not None:
                return self._merge_with_known(platform_variants)

        # fallback：本地扫描
        return self._scan_locally()

    def scan_variants(self) -> list[VSRVariantInfo]:
        """同步扫描接口（fallback 本地扫描）。"""
        return self._scan_locally()

    def _merge_with_known(self, platform_variants: list[dict]) -> list[VSRVariantInfo]:
        """合并平台 variant 数据 + _KNOWN_VARIANTS 引擎特有字段。"""
        known_lookup: dict[str, dict] = {v["name"]: v for v in _KNOWN_VARIANTS}

        result: list[VSRVariantInfo] = []
        for pv in platform_variants:
            v_name = pv.get("name", "")
            kv = known_lookup.get(v_name, {})
            available = bool(pv.get("available", False))

            result.append(VSRVariantInfo(
                name=v_name,
                base_model=pv.get("base_model", kv.get("base_model", "")),
                precision=pv.get("precision", kv.get("precision", "fp32")),
                loader=pv.get("loader", kv.get("loader", "sttn")),
                vram_required=pv.get("vram_required", kv.get("vram_required", "未知")),
                available=available,
                inpaint_mode=kv.get("inpaint_mode", "sttn-auto"),
                requires_ocr=kv.get("requires_ocr", False),
            ))
            if available:
                self._logger.info("发现模型变体: %s (%s)", v_name, pv.get("alias", ""))

        if not any(pv.get("available") for pv in platform_variants):
            self._logger.warning(
                "未在 %s 中找到任何可用的 VSR 模型变体，"
                "请到模型管理界面下载（STTN：Dev-Jahn/sttn-auto|sttn-det，"
                "LAMA：Sanster/models，ProPainter：camenduru/ProPainter）",
                self.model_dir,
            )

        return result

    def _scan_locally(self) -> list[VSRVariantInfo]:
        """fallback 本地扫描。

        本地 fallback 策略（遵循 project-rules.md 第二部分 §5）：
            - 只检查标志文件（marker_files）存在性
            - 不严格校验大小（避免断点续传残文件被误判可用）
            - 让 variant 在 UI 中可见，触发下载入口
        """
        capabilities: list[VSRVariantInfo] = []

        if not self.model_dir.exists():
            self._logger.warning("模型目录不存在: %s", self.model_dir)
            return capabilities

        # OCR 组标志（server/mobile 任一存在即可）
        has_ocr = self._check_ocr_markers()

        for variant_def in _KNOWN_VARIANTS:
            variant_dir = self.model_dir / variant_def["subdir"]

            if not variant_dir.exists():
                self._logger.debug(
                    "变体 %s 目录不存在: %s", variant_def["name"], variant_dir,
                )
                continue

            # 本地 fallback 只检查标志文件
            has_model = self._check_variant_files(variant_def["name"])
            if not has_model:
                self._logger.debug(
                    "变体 %s 标志文件缺失: %s", variant_def["name"], variant_dir,
                )
                continue

            # 需要 OCR 的算法：OCR 组缺失时 variant 不可用（提示下载）
            if variant_def["requires_ocr"] and not has_ocr:
                self._logger.warning(
                    "变体 %s 需要 OCR 检测模型但未下载（vsr-ocr-server/mobile），"
                    "请到模型管理界面下载任一 OCR 模型",
                    variant_def["name"],
                )
                continue

            cap = VSRVariantInfo(
                name=variant_def["name"],
                base_model=variant_def["base_model"],
                precision=variant_def["precision"],
                loader=variant_def["loader"],
                vram_required=variant_def["vram_required"],
                available=True,
                inpaint_mode=variant_def["inpaint_mode"],
                requires_ocr=variant_def["requires_ocr"],
            )
            capabilities.append(cap)
            self._logger.info(
                "发现模型变体: %s (%s, inpaint_mode=%s)",
                cap.name, variant_def["alias"], variant_def["inpaint_mode"],
            )

        if not capabilities:
            self._logger.warning(
                "未在 %s 中找到任何 VSR 模型变体，"
                "请到模型管理界面下载（STTN：Dev-Jahn/sttn-auto|sttn-det，"
                "LAMA：Sanster/models，ProPainter：camenduru/ProPainter）",
                self.model_dir,
            )

        return capabilities

    def _check_variant_files(self, variant_name: str) -> bool:
        """检查 variant 所需的标志文件是否存在（本地 fallback）。"""
        variant_def = None
        for v in _KNOWN_VARIANTS:
            if v["name"] == variant_name:
                variant_def = v
                break
        if variant_def is None:
            return False

        variant_dir = self.model_dir / variant_def["subdir"]
        for marker in variant_def.get("marker_files", []):
            if not (variant_dir / marker).exists():
                return False
        return True

    def _check_ocr_markers(self) -> bool:
        """检查 OCR 组标志（server/mobile 任一存在即可，exclusive_group 语义）。"""
        for ocr_subdir, marker in _OCR_MARKERS:
            if (self.model_dir / ocr_subdir / marker).exists():
                return True
        return False

    def resolve_variant_path(self, variant_name: str) -> Optional[Path]:
        """根据变体名解析模型路径。

        Args:
            variant_name: 变体名（如 VSR/STTN-AUTO）

        Returns:
            模型目录路径（如 models/video/vsr/sttn-auto），不存在返回 None
        """
        for variant_def in _KNOWN_VARIANTS:
            if variant_def["name"] == variant_name:
                path = self.model_dir / variant_def["subdir"]
                return path if path.exists() else None
        return None

    def get_variant_definition(self, variant_name: str) -> Optional[dict]:
        """获取 variant 的完整定义。

        供 wrapper.load() 用于文件完整性校验。

        Args:
            variant_name: variant 名称

        Returns:
            variant 定义字典，不存在返回 None
        """
        for v in _KNOWN_VARIANTS:
            if v["name"] == variant_name:
                return v
        return None
