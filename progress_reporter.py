"""VSR 视频字幕移除 SSE 进度上报器。

把 SubtitleRemover 的处理过程转换为 AIReliable 进度帧，
并通过 SDK PlatformTaskReporter 同步任务状态到 gateway（v0.3.4+）。

进度阶段（与 vsr_wrapper.VSRProgressStage 对齐）：
    start               0.00          任务创建 / 开始
    preparing           0.05          参数准备（视频尺寸、选区换算）
    processing          0.05 - 0.90   OCR 检测 + 逐帧修复（progress_total 0-100 映射）
    merging             0.90 - 0.95   合并原音频
    completed           1.00          完成（含 video_url）
    error               -             失败
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from aireliable_engine import (
    SSEStream,
    get_platform_task_reporter,
    make_error_frame,
    make_final_frame,
    make_frame,
    PlatformTaskReporter,
)

from .vsr_wrapper import VSRProgressStage

logger = logging.getLogger(__name__)


# 进度阶段（复用 vsr_wrapper 的定义，避免双源漂移）
ProgressStage = VSRProgressStage


class ProgressReporter:
    """VSR 进度上报器。

    封装 SSEStream，提供按阶段构造帧的便捷方法。
    v0.3.4+ 同时通过 SDK PlatformTaskReporter 上报任务状态到 gateway。
    """

    def __init__(
        self,
        stream: SSEStream,
        task_reporter: Optional[PlatformTaskReporter] = None,
    ):
        """初始化进度上报器。

        Args:
            stream: SSE 流
            task_reporter: 可选的 SDK 任务上报器（v0.3.4+）。
                为 None 时不上报任务状态到 gateway（仅 SSE 推送）。
        """
        self.stream = stream
        self._start_time: float = time.time()
        self._task_reporter: Optional[PlatformTaskReporter] = task_reporter
        self._task_id: Optional[str] = None

    @property
    def elapsed(self) -> float:
        """已用时（秒）。"""
        return time.time() - self._start_time

    def set_task_id(self, task_id: str) -> None:
        """设置任务 ID（用于 SDK 上报）。"""
        self._task_id = task_id

    async def init_task_reporter(self) -> None:
        """v0.3.4+ 异步初始化 SDK 任务上报器（捕获主 loop + 获取单例）。

        必须在 emit_* 方法之前调用（在 _infer_impl 入口处）。
        失败不阻断推理。
        """
        if self._task_reporter is None and self._task_id:
            try:
                self._task_reporter = await get_platform_task_reporter()
            except Exception as e:
                logger.warning("init_task_reporter 失败（不阻断推理）: %s", e)

    async def report_created(
        self,
        request_summary: dict,
        variant: str = "",
        task_type: str = "",
    ) -> None:
        """v0.3.4+ 上报任务创建到 gateway（status=queued）。"""
        if self._task_reporter and self._task_id:
            try:
                await self._task_reporter.report_created(
                    task_id=self._task_id,
                    request_summary=request_summary,
                    variant=variant,
                    task_type=task_type,
                )
            except Exception as e:
                logger.warning("report_created 失败（不阻断推理）: %s", e)

    def _fire_running(self, stage: str) -> None:
        """fire-and-forget 上报任务进入 running 状态。"""
        if self._task_reporter and self._task_id:
            self._task_reporter.fire_report_running(self._task_id, stage=stage)

    def _fire_progress(self, progress: float, stage: str) -> None:
        """fire-and-forget 上报任务进度。"""
        if self._task_reporter and self._task_id:
            self._task_reporter.fire_report_progress(
                self._task_id, progress=progress, stage=stage,
            )

    def _fire_succeeded(self, result: dict) -> None:
        """fire-and-forget 上报任务成功。"""
        if self._task_reporter and self._task_id:
            self._task_reporter.fire_report_succeeded(self._task_id, result=result)

    def _fire_failed(self, error: dict) -> None:
        """fire-and-forget 上报任务失败。"""
        if self._task_reporter and self._task_id:
            self._task_reporter.fire_report_failed(self._task_id, error=error)

    # ===== 通用帧 =====

    def emit(self, progress: float, stage: str, **extra: Any) -> None:
        """发送一帧进度（线程安全）。

        SubtitleRemover 的进度监听器在工作线程中回调，因此使用
        put_threadsafe（与 stream 绑定的主事件循环线程安全）。
        """
        frame = make_frame(progress, stage, **extra)
        self.stream.put_threadsafe(frame)
        # 同步上报进度到 gateway（fire-and-forget，不阻塞 SSE 流）
        self._fire_progress(progress, stage)

    async def emit_async(self, progress: float, stage: str, **extra: Any) -> None:
        """发送一帧进度（协程内调用）。"""
        frame = make_frame(progress, stage, **extra)
        await self.stream.put(frame)
        self._fire_progress(progress, stage)

    def emit_error(self, message: str, error_type: str = "video_subtitle_error") -> None:
        """发送错误帧。"""
        frame = make_error_frame(message, error_type)
        self.stream.put_threadsafe(frame)
        self._fire_failed({"type": error_type, "message": message})

    def emit_heartbeat(self) -> None:
        """发送心跳帧（重置 gateway httpx read timeout）。

        视频处理耗时长（单条视频可能数分钟），期间若 SubtitleRemover
        长时间无进度回调，会触发 gateway read timeout → 502 Bad Gateway。
        在长时间阻塞期间定期推送心跳帧，重置 gateway 的 read timeout 计时。
        """
        self.emit(
            0.50,
            "video_subtitle.heartbeat",
            message="处理中（心跳）",
            elapsed=self.elapsed,
        )

    # ===== 阶段便捷方法 =====

    def emit_start(self, task_id: str, description: str = "") -> None:
        """开始处理。"""
        self._task_id = task_id
        self.emit(
            0.0,
            ProgressStage.START,
            message="开始视频字幕/水印移除",
            task_id=task_id,
            description=description[:200],
        )
        self._fire_running(ProgressStage.START)

    def emit_preparing(self) -> None:
        """参数准备（视频尺寸 / 选区换算）。"""
        self.emit(0.05, ProgressStage.PREPARING, message="参数准备中…")

    def emit_processing(self, progress: float, detail: str = "") -> None:
        """处理中（OCR 检测 + 逐帧修复）。"""
        self.emit(
            max(0.05, min(0.90, progress)),
            ProgressStage.PROCESSING,
            message=detail or f"处理中 {int(progress * 100)}%",
        )

    def emit_merging(self) -> None:
        """合并原音频。"""
        self.emit(0.92, ProgressStage.MERGING, message="合并原音频…")

    async def emit_completed(self, task_id: str, result: dict) -> None:
        """完成。"""
        timing = {
            "total_seconds": self.elapsed,
        }
        final_payload = {
            "task_id": task_id,
            **result,
        }
        frame = make_final_frame(final_payload, timing)
        await self.stream.put(frame)
        self._fire_succeeded(final_payload)
