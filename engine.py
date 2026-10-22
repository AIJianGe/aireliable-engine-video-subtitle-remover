"""VSR 视频字幕/水印移除引擎推理核心。

继承 ``CustomEngine``（全自定义模式 / integrated 范式），实现 7 个
``_xxx_impl`` 方法，封装 VSRWrapper 的加载/推理/卸载流程。

8 端点契约：
    - /capabilities  返回扫描到的 VSR 模型变体（4 算法，含 OCR 依赖信息）
    - /load          校验模型文件 + 配置 inpaint 算法（VSR 模型惰性加载，快速返回）
    - /unload        释放模型缓存 + GPU 显存
    - /infer         解析 VSRInputs → wrapper.run_subtitle_removal → SSE 推送
    - /status        返回引擎状态（已加载变体 / 设备 / inpaint 模式）
    - /lora/load     v1 不支持 LoRA（预留接口，M2 扩展项）
    - /lora/unload   v1 不支持 LoRA（预留接口，M2 扩展项）

任务类型：video_subtitle_remover（硬字幕/水印移除，保留原音频）。

v0.2.11+ 方案 B：variant 扫描优先走平台 API（engine.get_variants()），
平台不可达时 fallback 到本地文件扫描。
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from aireliable_engine import (
    CapabilitiesResponse,
    CustomEngine,
    EngineBusyError,
    InferRequest,
    LoadRequest,
    LoadResponse,
    LoraLoadResponse,
    LoraUnloadResponse,
    SSEStream,
    StatusResponse,
    UnloadResponse,
    VariantInfo,
)
from aireliable_engine.state_machine import RuntimeState

from .config import EngineConfig
from .model_scanner import ModelScanner
from .progress_reporter import ProgressReporter
from .schemas import TASK_TYPE, VSRInputs, VSRVariantInfo
from .vsr_wrapper import VSRProgressStage, VSRWrapper

logger = logging.getLogger(__name__)


class VSRSubtitleRemoverEngine(CustomEngine):
    """VSR 视频字幕/水印移除引擎。

    全自定义模式（integrated 范式），直接控制模型加载/推理流程。
    支持 4 种 inpaint 算法 variant（STTN-AUTO / STTN-DET / LAMA / PROPAINTER），
    PP-OCRv5 检测字幕区域（STTN-DET / LAMA / PROPAINTER 依赖）。
    """

    def __init__(self, config: EngineConfig):
        super().__init__(
            engine_id=config.engine_id,
            engine_name=config.engine_name,
            capability=config.capability,
            engine_type=config.engine_type,
        )
        self.config: EngineConfig = config

        # ===== 核心组件 =====
        self.wrapper = VSRWrapper(
            models_root=config.models_root,
            model_dir=config.model_dir,
            output_root=config.output_root,
            logger=logger,
        )
        # v0.2.11+ 方案 B：注入 engine=self，启用平台 API 声明式检测
        self.scanner = ModelScanner(
            models_root=config.models_root,
            model_dir=config.model_dir,
            engine=self,
            logger=logger,
        )

        # 启动时扫描模型目录（同步初始化为空，异步扫描在 lifespan 中触发）
        self._available_variants: list[VSRVariantInfo] = []
        self._refresh_scan()

        # 输出根目录（供 BaseEngine.get_output_path 使用）
        self.output_root = config.output_root

        # 读取 manifest 中的 default_variant（供 BaseEngine.infer auto-load 使用）
        # 优先级：request.variant > self.default_variant > 第一个 available variant
        _manifest_path = Path(__file__).parent / "manifest.yaml"
        self.default_variant = self._read_default_variant_from_manifest(_manifest_path)

        # 任务类型（任务中心筛选语义）
        self.task_type = TASK_TYPE

    # ============ 扫描 ============

    def _refresh_scan(self) -> None:
        """刷新模型扫描结果（同步 fallback，用于 __init__ 阶段）。

        v0.2.11+ 方案 B：此方法走 fallback 本地扫描逻辑。
        平台 API 异步扫描请调用 `_refresh_scan_async()`（lifespan startup 阶段调用）。
        """
        logger.info("刷新模型扫描（本地 fallback）: %s", self.config.model_dir)
        self._available_variants = self.scanner.scan_variants()

    async def _refresh_scan_async(self) -> None:
        """v0.2.11+ 方案 B：异步刷新模型扫描结果（优先平台 API）。

        在 lifespan startup 阶段调用，此时平台 API 已就绪。
        平台不可达时 fallback 到本地扫描。
        """
        logger.info("异步刷新模型扫描（方案 B 平台 API）: %s", self.config.model_dir)
        try:
            self._available_variants = await self.scanner.scan_variants_async()
        except Exception as e:
            logger.warning("异步扫描失败，回退到本地扫描: %s", e)
            self._available_variants = self.scanner.scan_variants()

    # ============ 7 个 _xxx_impl 实现 ============

    async def _capabilities_impl(self) -> CapabilitiesResponse:
        """GET /capabilities：返回可用模型变体（4 算法）。"""
        variants: list[VariantInfo] = []
        for v in self._available_variants:
            variants.append(VSRVariantInfo(
                name=v.name,
                base_model=v.base_model,
                precision=v.precision,
                loader=v.loader,
                vram_required=v.vram_required,
                available=v.available,
                inpaint_mode=v.inpaint_mode,
                requires_ocr=v.requires_ocr,
            ))

        return CapabilitiesResponse(
            engine_id=self.engine_id,
            engine_name=self.engine_name,
            capability=self.capability,
            type=self.engine_type,
            variants=variants,
            loras=[],  # v1 不支持 LoRA
            supported_sizes=["1920x1080"],  # 建议最大输入分辨率
            default_variant=self.default_variant,
            # v0.4.1+ MCP 化：推理输入 JSON Schema
            input_schema=VSRInputs.model_json_schema(),
            # 单能力引擎：capabilities 列表退化为 [capability]
            capabilities=[self.capability],
        )

    async def _load_impl(self, variant: Optional[str]) -> LoadResponse:
        """POST /load：校验模型文件 + 配置 inpaint 算法（快速返回）。

        VSR 模型为惰性加载（权重在 SubtitleRemover.run() 内部按需加载），
        此方法只做：
            1. variant 路径解析（不存在抛 FileNotFoundError）
            2. 模型文件完整性校验（缺失抛 RuntimeError 指引下载）
            3. 算法 / OCR / 设备配置（backend config）
        """
        if not variant:
            raise RuntimeError("未指定 variant，无法加载")

        variant_path = self.scanner.resolve_variant_path(variant)
        if variant_path is None:
            raise FileNotFoundError(
                f"variant 不存在: {variant} "
                f"（可用: {[v.name for v in self._available_variants]}）"
            )

        device = self.config.resolve_device()
        detect_mode = self._resolve_detect_mode(variant)

        start_time = time.time()
        try:
            await asyncio.to_thread(
                self.wrapper.load,
                variant,
                variant_path,
                device,
                detect_mode,
            )
        except Exception as e:
            logger.exception("VSR 模型配置失败: variant=%s", variant)
            raise RuntimeError(f"VSR 模型配置失败: {e}") from e

        load_time = time.time() - start_time
        logger.info(
            "加载就绪: variant=%s, inpaint_mode=%s, device=%s, load_time=%.2fs",
            variant, self.wrapper.loaded_inpaint_mode, device, load_time,
        )

        return LoadResponse(
            variant=variant,
            state="ready",
            vram_used="0MB",  # VSR 惰性加载，未实际占用显存
            load_time=load_time,
        )

    def _resolve_detect_mode(self, variant: str) -> Optional[str]:
        """根据 variant 选择 OCR 检测模式。

        - STTN-AUTO：无需 OCR，返回 None
        - STTN-DET / LAMA / PROPAINTER：优先 mobile（轻量），
          若 mobile 未下载则回退 server
        """
        vdef = self.scanner.get_variant_definition(variant)
        if vdef is None or not vdef.get("requires_ocr"):
            return None
        mobile_marker = self.config.model_dir / "V5" / "ch_det_fast" / "inference.yml"
        if mobile_marker.exists():
            return "PP_OCRv5_MOBILE"
        return "PP_OCRv5_SERVER"

    async def _unload_impl(self) -> UnloadResponse:
        """POST /unload：释放模型缓存 + GPU 显存。

        即使 wrapper 未加载模型（如 ERROR 状态下），
        也要确保状态机回到 IDLE，避免 ERROR 状态卡死无法恢复。
        """
        if self.state_machine.state == RuntimeState.ERROR:
            try:
                self.state_machine.reset()
                logger.info("unload()：状态机从 ERROR 重置为 IDLE")
            except Exception as e:
                logger.warning("unload()：reset 状态机失败: %s", e)

        if not self.wrapper.is_loaded:
            if self.state_machine.state != RuntimeState.IDLE:
                try:
                    self.state_machine.transition(RuntimeState.IDLE)
                except Exception as e:
                    logger.warning("unload()：状态转换到 IDLE 失败: %s", e)
            return UnloadResponse(state="idle", vram_freed="0MB")

        await asyncio.to_thread(self.wrapper.unload)

        if self.state_machine.state != RuntimeState.IDLE:
            try:
                self.state_machine.transition(RuntimeState.IDLE)
            except Exception as e:
                logger.warning("unload()：unload 后状态转换到 IDLE 失败: %s", e)

        logger.info("卸载完成（VSR 模型缓存 + GPU 显存已释放）")
        return UnloadResponse(state="idle", vram_freed="0MB")

    async def _infer_impl(self, request: InferRequest, stream: SSEStream) -> None:
        """POST /infer：视频字幕/水印移除（SSE 流式推送进度）。

        流程：
            1. 生成 task_id（若 request.task_id 为 None）
            2. 解析 VSRInputs（从 request.inputs）
            3. 解析目标视频路径（request.videos[].path 优先，兜底 inputs.video_path）
            4. 上报 start / preparing 帧
            5. 线程池执行 wrapper.run_subtitle_removal()（带心跳）
            6. 上报 completed 帧（含 video_url / output_path）

        错误处理：内部捕获异常，推送 error 帧后 return，
        不重新抛出（避免 BaseEngine.infer 把状态置为 ERROR）。
        """
        # ----- 0. 生成 task_id -----
        task_id = request.task_id or f"vsr_{uuid.uuid4().hex[:16]}"
        reporter = ProgressReporter(stream)

        # ----- 1. 解析输入 -----
        try:
            inputs = VSRInputs(**request.inputs)
        except Exception as e:
            logger.error("VSRInputs 解析失败: %s", e)
            reporter.emit_error(f"输入参数无效: {e}", "invalid_inputs")
            return

        # ----- 2. 上报任务创建到 gateway -----
        reporter.set_task_id(task_id)
        await reporter.init_task_reporter()

        # ----- 3. 解析目标视频路径 -----
        # 优先 request.videos[].path（网关直连链路），兜底 inputs.video_path
        video_path: Optional[str] = None
        if request.videos:
            video_path = request.videos[0].path
        if not video_path:
            video_path = inputs.video_path
        if not video_path:
            reporter.emit_error(
                "缺少目标视频：请通过 videos[].path 或 inputs.video_path 传入",
                "missing_video",
            )
            return
        if not Path(video_path).exists():
            reporter.emit_error(
                f"目标视频不存在: {video_path}",
                "video_not_found",
            )
            return

        # ----- 3.5 处理前自动加载：所选算法 ≠ 已加载模型时先切换 -----
        # WebUI / 平台任务中心直接传 variant（不显式调 /load）时，
        # 保证用户选择的算法真正执行，而不是沿用已加载的旧模型。
        # VSR 为惰性加载，wrapper.load() 仅做文件校验 + 配置，开销可忽略。
        requested_variant = (
            request.variant or self.wrapper.loaded_variant or self.default_variant
        )
        if requested_variant and self.wrapper.loaded_variant != requested_variant:
            logger.info(
                "处理前自动加载: 已加载=%s, 请求=%s → 切换",
                self.wrapper.loaded_variant, requested_variant,
            )
            reporter.emit(
                0.0,
                "model_loading",
                message=f"模型切换加载中（{requested_variant}）…",
            )
            try:
                await self._load_impl(requested_variant)
            except Exception as e:
                logger.exception(
                    "处理前自动加载失败: variant=%s", requested_variant,
                )
                reporter.emit_error(f"模型切换加载失败: {e}", "load_failed")
                return
            logger.info(
                "处理前自动加载完成: variant=%s, inpaint_mode=%s",
                requested_variant, self.wrapper.loaded_inpaint_mode,
            )

        await reporter.report_created(
            request_summary={
                "video": Path(video_path).name,
                "inpaint_mode": self.wrapper.loaded_inpaint_mode or "sttn-auto",
                "detect_mode": inputs.detect_mode,
                "sub_areas": inputs.sub_areas,
            },
            variant=request.variant or self.wrapper.loaded_variant or "",
            task_type=self.task_type,
        )

        # ----- 4. 上报开始阶段 -----
        reporter.emit_start(task_id, Path(video_path).name)
        reporter.emit_preparing()

        # ----- 5. 构造入参（dict 形式传给 wrapper）-----
        infer_inputs: dict[str, Any] = {
            "video_path": video_path,
            "sub_areas": inputs.sub_areas,
            "ab_sections": inputs.ab_sections,
            "detect_mode": inputs.detect_mode,
            "params": inputs.params,
        }

        # 进度回调（工作线程调用 → put_threadsafe 线程安全）
        def _progress(progress: float, stage: str, detail: str) -> None:
            if stage == VSRProgressStage.PROCESSING:
                reporter.emit_processing(progress, detail)
            else:
                reporter.emit(progress, stage, message=detail)

        # 心跳任务：run_subtitle_removal() 是同步阻塞调用，期间定期推送心跳帧，
        # 重置 gateway 的 read timeout 计时。
        async def _heartbeat():
            elapsed = 0
            while True:
                await asyncio.sleep(30)
                elapsed += 30
                reporter.emit_heartbeat()

        heartbeat_task = asyncio.create_task(_heartbeat())
        try:
            result = await asyncio.to_thread(
                self.wrapper.run_subtitle_removal,
                infer_inputs,
                _progress,
            )
        except Exception as e:
            logger.exception("VSR 推理失败")
            reporter.emit_error(f"字幕/水印移除失败: {e}", "infer_failed")
            return
        finally:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass

        # ----- 6. 合并音频阶段 + 上报完成 -----
        reporter.emit_merging()
        final_result = {
            "video_url": result["video_url"],
            "output_path": result["output_path"],
            "elapsed": result["elapsed"],
            "variant": self.wrapper.loaded_variant,
            "inpaint_mode": self.wrapper.loaded_inpaint_mode,
            "device": self.wrapper._device,
            "detect_mode": inputs.detect_mode,
            "input_video": Path(video_path).name,
        }

        logger.info(
            "VSR 处理完成: task_id=%s, video_url=%s, elapsed=%.1fs",
            task_id, result["video_url"], result["elapsed"],
        )

        await reporter.emit_completed(task_id, final_result)

    async def _status_impl(self) -> StatusResponse:
        """GET /status：返回引擎状态。"""
        raw_device = self.wrapper._device
        if raw_device.startswith("cuda"):
            device_name = "GPU (CUDA)"
        elif raw_device == "cpu":
            device_name = "CPU"
        else:
            device_name = raw_device
        return StatusResponse(
            state=self.state_machine.state.value,
            loaded_variant=self.wrapper.loaded_variant,
            vram_used="0MB",  # VSR 惰性加载，显存占用无法精确统计
            loaded_loras=[],  # v1 不支持 LoRA
            last_used_at=self.state_machine.last_used_at,
            uptime=self.state_machine.uptime,
            extra={
                "device": raw_device,
                "device_name": device_name,
                "is_loaded": self.wrapper.is_loaded,
                "loaded_inpaint_mode": self.wrapper.loaded_inpaint_mode,
                "task_type": self.task_type,
                "available_variants": [
                    {
                        "name": v.name,
                        "available": v.available,
                        "inpaint_mode": v.inpaint_mode,
                        "requires_ocr": v.requires_ocr,
                        "vram_required": v.vram_required,
                    }
                    for v in self._available_variants
                ],
            },
        )

    # ============ LoRA（v1 预留）============

    async def _load_lora_impl(self, name: str, weight: float) -> LoraLoadResponse:
        """POST /lora/load：VSR v1 不支持 LoRA（预留接口，后续版本扩展）。"""
        raise RuntimeError(
            "VSR v1 不支持 LoRA（预留接口，后续版本扩展）"
        )

    async def _unload_lora_impl(self, name: str) -> LoraUnloadResponse:
        """POST /lora/unload：VSR v1 不支持 LoRA（预留接口，后续版本扩展）。"""
        raise RuntimeError(
            "VSR v1 不支持 LoRA（预留接口，后续版本扩展）"
        )
