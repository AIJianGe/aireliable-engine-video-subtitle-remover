"""VSR SubtitleRemover 服务化封装。

把上游 ``backend/main.py`` 的 ``SubtitleRemover``（桌面 GUI 场景）
包装为可被平台引擎调用的服务接口：

    - ``load()``：文件完整性校验 + 算法参数配置（VSR 模型为惰性加载，
      实际权重在 ``SubtitleRemover.run()`` 内部按需加载，load 只做
      校验与模式设置，保证 /load 快速返回）
    - ``run_subtitle_removal()``：在线程中执行 ``SubtitleRemover.run()``，
      进度监听器 → SSE 进度帧回调
    - ``unload()``：释放缓存模型（gc + torch.empty_cache）

关键服务化改造：
    1. 输出路径重定向到 ``output_root/video/``（平台 /output 挂载目录）
    2. stdout 临时重定向 → logging（tqdm 进度条 / print 输出转日志，
       不污染 core 的 JSON stdout 采集；logging handler 持有原始
       sys.stdout 引用，JSON 输出不受影响）
    3. 归一化选区（0-1）→ 像素坐标（上游格式）
    4. 设备选择：CPU 时关闭硬件加速（config.hardwareAcceleration=False），
       与平台设备下拉语义对齐
"""

from __future__ import annotations

import gc
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

import cv2

logger = logging.getLogger(__name__)


# ===== 进度阶段定义（与 WebUI 展示语义一致）=====
class VSRProgressStage:
    START = "video_subtitle.start"          # 0.00
    PREPARING = "video_subtitle.preparing"  # 0.05
    PROCESSING = "video_subtitle.processing"  # 0.05 - 0.90（OCR 检测 + 逐帧修复）
    MERGING = "video_subtitle.merging"      # 0.90 - 0.95（合并原音频）
    COMPLETED = "video_subtitle.completed"  # 1.00
    ERROR = "video_subtitle.error"


class _StdoutRedirect:
    """把 stdout 的 print / tqdm 输出转发到 Python logging。

    仅替换 ``sys.stdout``（print / tqdm.write 走当前引用），
    logging 的 StreamHandler 持有创建时的原始 stdout 引用，
    因此 JSON 日志输出不受影响（无递归）。
    """

    def __init__(self, log_func: Callable[[str], None]):
        self._log = log_func

    def write(self, data: str) -> int:
        data = data.rstrip("\n")
        if data.strip():
            self._log(data)
        return len(data)

    def flush(self) -> None:
        pass


class VSRWrapper:
    """SubtitleRemover 服务化封装。"""

    def __init__(
        self,
        models_root: Path,
        model_dir: Path,
        output_root: Path,
        logger: Optional[logging.Logger] = None,
    ):
        self.models_root = models_root
        self.model_dir = model_dir
        self.output_root = output_root
        self._logger = logger or logging.getLogger(__name__)

        # 加载状态（VSR 模型惰性加载，is_loaded 表示配置已就绪）
        self.is_loaded = False
        self.loaded_variant: Optional[str] = None
        self.loaded_inpaint_mode: Optional[str] = None
        self._device = "cpu"

    # ============ 文件完整性校验 ============

    def _check_variant_files(self, variant: str, variant_path: Path) -> None:
        """校验 variant 所需模型文件存在（缺失时抛 RuntimeError 指引下载）。

        Args:
            variant: variant 名称（如 VSR/STTN-AUTO）
            variant_path: variant 模型目录（如 models/video/vsr/sttn-auto）

        Raises:
            RuntimeError: 模型文件缺失 / 目录不存在
        """
        from .model_scanner import ModelScanner

        vdef = ModelScanner(self.models_root, self.model_dir).get_variant_definition(variant)
        if vdef is None:
            raise RuntimeError(f"未知 variant: {variant}")

        missing: list[str] = []
        for marker in vdef.get("marker_files", []):
            if not (variant_path / marker).exists():
                missing.append(str(variant_path / marker))

        # 需要 OCR 的算法：检查 OCR 组任一存在
        if vdef.get("requires_ocr"):
            ocr_server = self.model_dir / "V5" / "ch_det" / "inference.yml"
            ocr_mobile = self.model_dir / "V5" / "ch_det_fast" / "inference.yml"
            if not (ocr_server.exists() or ocr_mobile.exists()):
                missing.append("OCR 检测模型（video/vsr/V5/ch_det 或 ch_det_fast）")

        if missing:
            missing_display = "\n  ".join(missing)
            raise RuntimeError(
                f"模型文件不完整，缺失 {len(missing)} 项：\n"
                f"  {missing_display}\n\n"
                f"请到引擎详情页 → 模型管理 → 点击「下载缺失文件」按钮，"
                f"选择镜像源后下载。\n"
                f"本地路径：{self.model_dir}"
            )

    # ============ load / unload ============

    def load(
        self,
        variant: str,
        variant_path: Path,
        device: str = "auto",
        detect_mode: Optional[str] = None,
    ) -> None:
        """加载前校验 + 配置（VSR 模型惰性加载，此方法快速返回）。

        Args:
            variant: variant 名称
            variant_path: variant 模型目录
            device: 推理设备（auto / cpu / cuda:0）
            detect_mode: OCR 检测模式（PP_OCRv5_SERVER / PP_OCRv5_MOBILE）

        Raises:
            RuntimeError: 模型文件缺失
        """
        self._check_variant_files(variant, variant_path)

        # 解析设备：cpu → 关闭硬件加速（force CPU 语义）
        self._device = device
        from backend.config import config
        from backend.tools.constant import InpaintMode, SubtitleDetectMode

        # 1. 设置 inpaint 算法（variant → InpaintMode 枚举）
        vdef = self._get_variant_definition(variant)
        inpaint_mode_str = vdef["inpaint_mode"] if vdef else "sttn-auto"
        mode_enum = InpaintMode(inpaint_mode_str)  # 非法值抛 ValueError
        config.inpaintMode.value = mode_enum

        # 2. 设置 OCR 检测模式（可选，默认 server）
        if detect_mode:
            config.subtitleDetectMode.value = SubtitleDetectMode(detect_mode)

        # 3. 设备：cpu 强制关闭硬件加速（HardwareAccelerator.set_enabled）
        from backend.tools.hardware_accelerator import HardwareAccelerator

        use_accel = not self._device.startswith("cpu")
        HardwareAccelerator.instance().set_enabled(use_accel)
        config.hardwareAcceleration.value = use_accel

        self.is_loaded = True
        self.loaded_variant = variant
        self.loaded_inpaint_mode = inpaint_mode_str
        self._logger.info(
            "VSR 加载就绪: variant=%s, inpaint_mode=%s, device=%s, detect_mode=%s",
            variant, inpaint_mode_str, self._device, detect_mode or "server",
        )

    def unload(self) -> None:
        """卸载：释放缓存模型 + GPU 缓存。"""
        self.is_loaded = False
        self.loaded_variant = None
        self.loaded_inpaint_mode = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        self._logger.info("VSR 已卸载（模型缓存释放 + GPU 缓存清空）")

    # ============ 推理 ============

    def run_subtitle_removal(
        self,
        inputs: dict[str, Any],
        progress_cb: Optional[Callable[[float, str, str], None]] = None,
    ) -> dict[str, Any]:
        """执行视频字幕/水印移除（在调用线程中同步运行）。

        Args:
            inputs: 推理入参（schemas.VSRInputs 的 dict 形式）
                - video_path: 目标视频绝对路径
                - sub_areas: 归一化选区 [[ymin,ymax,xmin,xmax], ...]（空=全屏）
                - ab_sections: AB 段时间区间（可选）
                - detect_mode: OCR 检测模式（可选）
                - params: 算法参数透传（可选）
            progress_cb: 进度回调 (progress 0-1, stage, detail)（可选）

        Returns:
            dict: {"output_path": ..., "video_url": ..., "elapsed": ...}

        Raises:
            RuntimeError: 视频无效 / 处理失败（含缺失模型指引）
        """
        from backend.config import config
        from backend.tools.constant import SubtitleDetectMode
        from backend.tools.common_tools import is_video_or_image

        start_ts = time.time()
        video_path = inputs.get("video_path")
        if not video_path or not Path(video_path).exists():
            raise RuntimeError(f"目标视频不存在: {video_path}")
        if not is_video_or_image(video_path):
            raise RuntimeError(f"不支持的文件类型（需视频或图片）: {video_path}")

        # 1. 参数透传（覆盖 config 默认值）
        self._apply_params(inputs.get("params") or {})

        # 2. OCR 检测模式（若请求指定）
        detect_mode = inputs.get("detect_mode")
        if detect_mode:
            config.subtitleDetectMode.value = SubtitleDetectMode(detect_mode)

        # 3. 视频尺寸（归一化选区 → 像素坐标）
        cap = cv2.VideoCapture(video_path)
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
        cap.release()
        if frame_h <= 0 or frame_w <= 0:
            raise RuntimeError(f"无法读取视频尺寸: {video_path}")

        sub_areas_pixel = self._to_pixel_coords(
            inputs.get("sub_areas") or [], frame_h, frame_w
        )

        # 4. 构造 SubtitleRemover（延迟 import，首次调用加载 backend 依赖）
        from backend.main import SubtitleRemover

        if progress_cb:
            progress_cb(0.05, VSRProgressStage.PREPARING, "准备处理中…")

        remover = SubtitleRemover(video_path, gui_mode=False)
        remover.sub_areas = sub_areas_pixel

        ab_sections = inputs.get("ab_sections")
        if ab_sections:
            # 上游 ab_sections 格式：[range(start, end), ...]（is_frame_number_in_ab_sections 依赖）
            remover.ab_sections = [
                range(int(start), int(end)) for start, end in ab_sections
            ]

        # 5. 输出路径重定向到平台 output/video/（/output 静态挂载目录）
        vd_name = Path(video_path).stem
        out_dir = self.output_root / "video"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{vd_name}_no_sub_{uuid.uuid4().hex[:8]}.mp4"
        remover.video_out_path = str(out_path)

        # 6. 进度监听 → 回调（progress_total 0-100 → 0.05-0.90）
        def _on_progress(p_total: int, is_finished: bool):
            if progress_cb is None:
                return
            progress = 0.05 + 0.85 * (max(0, min(100, int(p_total))) / 100.0)
            progress_cb(progress, VSRProgressStage.PROCESSING, f"处理中 {int(p_total)}%")

        remover.add_progress_listener(_on_progress)

        # 7. 抑制 tqdm 进度条 + stdout 重定向（print/tqdm.write → logging）
        remover._tqdm_disable = True  # main.py run() 中 disable tqdm 渲染

        def _log_stdout(line: str) -> None:
            logger.info("[VSR] %s", line)

        old_stdout = sys.stdout
        sys.stdout = _StdoutRedirect(_log_stdout)
        try:
            try:
                remover.run()
            except Exception as e:
                logger.exception("VSR 处理失败: %s", video_path)
                raise RuntimeError(f"字幕/水印移除失败: {e}") from e
        finally:
            sys.stdout = old_stdout

        # 8. 校验输出
        if not out_path.exists():
            raise RuntimeError(
                f"处理完成但未生成输出文件: {out_path}（请查看引擎日志）"
            )

        elapsed = time.time() - start_ts
        logger.info(
            "VSR 处理完成: %s -> %s（%.1fs, %d bytes）",
            video_path, out_path, elapsed, out_path.stat().st_size,
        )
        return {
            "output_path": str(out_path),
            "video_url": f"/output/video/{out_path.name}",
            "elapsed": round(elapsed, 2),
        }

    # ============ 工具方法 ============

    def _get_variant_definition(self, variant: str) -> Optional[dict]:
        from .model_scanner import ModelScanner

        return ModelScanner(self.models_root, self.model_dir).get_variant_definition(variant)

    def _to_pixel_coords(
        self, sub_areas: list[list[float]], frame_h: int, frame_w: int
    ) -> list[tuple[int, int, int, int]]:
        """归一化 [ymin,ymax,xmin,xmax]（0-1）→ 像素坐标 (ymin, ymax, xmin, xmax)。

        空列表返回空（SubtitleRemover.run 兜底为全屏区域）。
        """
        result: list[tuple[int, int, int, int]] = []
        for area in sub_areas:
            if len(area) != 4:
                self._logger.warning("跳过非法选区（需 4 个元素）: %s", area)
                continue
            ymin, ymax, xmin, xmax = area
            result.append((
                int(round(ymin * frame_h)),
                int(round(ymax * frame_h)),
                int(round(xmin * frame_w)),
                int(round(xmax * frame_w)),
            ))
        return result

    def _apply_params(self, params: dict[str, Any]) -> None:
        """把 infer 请求透传参数覆盖到 backend config（键名与 config 属性一致）。"""
        if not params:
            return
        from backend.config import config

        for key, value in params.items():
            attr = getattr(config, key, None)
            if attr is None:
                self._logger.warning("未知算法参数（忽略）: %s", key)
                continue
            attr.value = value
            self._logger.info("算法参数覆盖: %s = %s", key, value)
