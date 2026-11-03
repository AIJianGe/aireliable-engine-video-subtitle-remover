"""VSR 视频字幕/水印移除引擎 HTTP 服务。

8 个标准端点 / CORS / 日志端点 / 静态挂载 / 根路径跳转由 SDK 骨架
（``build_engine_app``）提供；本文件只保留引擎特有路由（``/upload`` + ``/first_frame``）。

启动：
    uv run --no-sync --project envs/video-vsr \\
        python -m engines.video_subtitle_remover_d1e2f3a4.server --port 7012

端点：
    GET  /health  /capabilities  /status          （骨架）
    POST /load  /unload  /infer(SSE)  /lora/*     （骨架；LoRA 为 v1 预留）
    GET  /output/*       静态视频文件（推理产物）
    GET|POST /log_level  （骨架）
    POST /upload         视频上传（WebUI，返回引擎侧路径）
    POST /first_frame    首帧提取（WebUI 涂抹选区用，返回 JPEG base64）
    GET  /ui/*           原生 WebUI 静态文件（webui/ 三件套）
    GET  /               307 重定向到 /ui/
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

# ===== sys.path 注入（必须先于任何 backend/ 相关导入）=====
# python -m engines.video_subtitle_remover_d1e2f3a4.server 以项目根为 cwd 启动，
# 默认只有项目根在 sys.path。以下两类导入需要额外路径：
#   - vsr_wrapper 的 `from backend.xxx import ...`        → 需要引擎根目录
#   - 本地 scenedetect 拷贝的顶层绝对导入（from scenedetect...）→ 需要 backend/ 目录
_ENGINE_ROOT = Path(__file__).resolve().parent
for _p in (_ENGINE_ROOT, _ENGINE_ROOT / "backend"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel

from aireliable_engine import build_engine_app, mount_upload, run_engine_server

from .config import EngineConfig, parse_args
from .engine import VSRSubtitleRemoverEngine

logger = logging.getLogger(__name__)

_WEBUI_DIR = _ENGINE_ROOT / "webui"


# ===== 全局引擎实例 =====
# 由 SDK 骨架的 startup_hook 绑定，供本文件的路由使用
_engine: Optional[VSRSubtitleRemoverEngine] = None
_config: Optional[EngineConfig] = None


def get_engine() -> VSRSubtitleRemoverEngine:
    """获取引擎实例。"""
    if _engine is None:
        raise RuntimeError("引擎未初始化（FastAPI lifespan 未执行？）")
    return _engine


def get_config() -> EngineConfig:
    """获取配置。"""
    if _config is None:
        raise RuntimeError("配置未初始化")
    return _config


# ===== Lifespan 钩子（由 SDK 骨架调用）=====

async def _on_startup(engine: VSRSubtitleRemoverEngine) -> None:
    """启动钩子：绑定全局实例 + 异步刷新 variant 扫描。"""
    global _engine

    _engine = engine
    # v0.2.11+ 方案 B：lifespan startup 阶段异步刷新扫描（平台 API 已就绪）
    # __init__ 阶段的同步 _refresh_scan 只是 fallback，此处用平台 API 重新检测
    await engine._refresh_scan_async()
    logger.info(
        "引擎实例化完成: state=%s, variants=%d, default_variant=%s",
        engine.state_machine.state.value,
        len(engine._available_variants),
        engine.default_variant,
    )


async def _on_shutdown(engine: VSRSubtitleRemoverEngine) -> None:
    """关闭钩子：卸载模型并清空全局引用。"""
    global _engine, _config

    logger.info("===== VSR 视频字幕/水印移除引擎关闭 =====")
    try:
        if engine.wrapper.is_loaded:
            try:
                await engine._unload_impl()
            except Exception as e:
                logger.warning("关闭时卸载模型失败: %s", e)
    finally:
        _engine = None
        _config = None


# ===== 引擎特有路由（WebUI 专属端点）=====

router = APIRouter()


class FirstFrameRequest(BaseModel):
    """首帧提取请求（WebUI 涂抹选区用）。

    Attributes:
        video_path: 引擎侧视频绝对路径（/upload 返回值）
        frame_no: 要提取的帧号（0 起；首帧黑场时可换一帧）
    """

    video_path: str
    frame_no: int = 0


mount_upload(
    router,
    lambda: get_config().output_root / "uploads",
    default_filename="upload.mp4",
    log_label="视频",
    url_prefix="/output/uploads",
)


@router.post("/first_frame")
async def first_frame(req: FirstFrameRequest) -> dict:
    """首帧提取（WebUI 涂抹选区用）。

    用 OpenCV 读取指定帧并编码为 JPEG base64 返回，与后端实际处理帧同源。
    不落盘文件（无清理负担），base64 体积 1080p 约 200-500KB，可接受。
    """
    import base64

    import cv2

    video_path = Path(req.video_path)
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"视频不存在: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise HTTPException(
            status_code=400,
            detail=f"无法打开视频（格式不支持或文件损坏）: {video_path}",
        )
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) + 0.5)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
        if frame_count <= 0 or width <= 0 or height <= 0:
            raise HTTPException(
                status_code=400, detail=f"无法读取视频尺寸: {video_path}"
            )
        # 帧号越界时钳制到有效范围
        frame_no = max(0, min(int(req.frame_no), max(0, frame_count - 1)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise HTTPException(
                status_code=400, detail=f"读取第 {frame_no} 帧失败: {video_path}"
            )
        ok, buf = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        )
        if not ok:
            raise HTTPException(status_code=500, detail="JPEG 编码失败")
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        logger.info(
            "首帧提取: %s (frame_no=%d, %dx%d, %d bytes)",
            video_path, frame_no, width, height, len(buf),
        )
        return {
            "image": f"data:image/jpeg;base64,{b64}",
            "width": width,
            "height": height,
            "frame_count": frame_count,
            "frame_no": frame_no,
        }
    finally:
        cap.release()


# ===== 创建 FastAPI 应用 =====

def create_app(config: EngineConfig | None = None) -> FastAPI:
    """创建 FastAPI 应用。

    Args:
        config: 引擎配置（None 则从命令行解析）

    Returns:
        FastAPI: 配置好的应用实例
    """
    global _config

    if config is None:
        config = parse_args()
    _config = config

    return build_engine_app(
        VSRSubtitleRemoverEngine,
        config=config,
        title="VSR Video Subtitle Remover Engine",
        description=(
            "AIReliable 平台的 VSR 视频字幕/水印移除引擎"
            "（PP-OCRv5 字幕检测 + 4 种 inpaint 算法：STTN-AUTO / STTN-DET / LAMA / ProPainter）"
        ),
        version="0.1.0",
        static_dirs={"/ui": _WEBUI_DIR, "/output": config.output_root},
        extra_routers=[router],
        startup_hook=_on_startup,
        shutdown_hook=_on_shutdown,
    )


# ===== 入口 =====

# 模块级 app 实例（uvicorn 直接引用）
app: Optional[FastAPI] = None


def main() -> None:
    """命令行启动入口。"""
    global app

    config = parse_args()
    app = create_app(config)
    run_engine_server(app, config=config)


if __name__ == "__main__":
    main()
