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

任务 71 P0-B：通用帧构造 / 任务上报 / 心跳 / 完成帧全部由 SDK
``BaseProgressReporter`` 提供，本文件只保留阶段常量与语义化方法。
"""

from __future__ import annotations

from aireliable_engine.progress import BaseProgressReporter

from .vsr_wrapper import VSRProgressStage


# 进度阶段（复用 vsr_wrapper 的定义，避免双源漂移）
ProgressStage = VSRProgressStage


class ProgressReporter(BaseProgressReporter):
    """VSR 进度上报器（阶段语义 + SDK 通用帧）。"""

    error_type = "video_subtitle_error"
    heartbeat_progress = 0.50
    heartbeat_stage = "video_subtitle.heartbeat"
    heartbeat_message = "处理中（心跳）"

    # ===== 阶段便捷方法 =====

    def emit_start(self, task_id: str, description: str = "") -> None:
        """开始处理。"""
        self.set_task_id(task_id)
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
        await super().emit_completed(
            {"task_id": task_id, **result},
            {"total_seconds": self.elapsed},
        )
