"""VSR 视频字幕/水印移除引擎 HTTP 服务（FastAPI）。

实现 8 个标准 HTTP 端点 + 挂载输出目录 + 原生 WebUI。

启动：
    uv run --no-sync --project envs/video-vsr \\
        python -m engines.video_subtitle_remover_d1e2f3a4.server --port 7012

端点：
    GET  /health         健康检查
    GET  /capabilities   能力声明（可用模型变体，4 算法）
    POST /load           校验模型文件 + 配置 inpaint 算法（VSR 惰性加载，快速返回）
    POST /unload         卸载模型（释放缓存 + 显存）
    POST /infer          视频字幕/水印移除推理（SSE 流式进度）
    GET  /status         状态查询（含已加载变体 / 设备 / inpaint 模式）
    POST /lora/load      LoRA 加载（v1 预留，不支持）
    POST /lora/unload    LoRA 卸载（v1 预留，不支持）
    GET  /output/*       静态视频文件（推理产物）
    GET  /log_level      查询/调整日志级别
    POST /upload         视频上传（WebUI，返回引擎侧路径）
    GET  /               307 重定向到 /ui/
    GET  /ui/*           原生 WebUI 静态文件（webui/ 三件套）
"""

from __future__ import annotations

import logging
import sys
import uuid
from contextlib import asynccontextmanager
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

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from aireliable_engine import (
    ErrorResponse,
    HealthResponse,
    InferRequest,
    LoadRequest,
    LoraLoadRequest,
    LoraUnloadRequest,
    register_log_level_endpoint,
    setup_engine_logging,
)

from .config import EngineConfig, parse_args
from .engine import VSRSubtitleRemoverEngine

logger = logging.getLogger(__name__)


# ===== 全局引擎实例 =====
# FastAPI 启动时初始化，所有端点共享
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


# ===== Lifespan 生命周期管理 =====

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 生命周期管理。

    启动时初始化引擎，关闭时清理资源（卸载模型）。
    """
    global _engine, _config

    # 解析配置（如果尚未从命令行注入）
    if _config is None:
        _config = parse_args()

    # 配置日志（v0.2.15+ 统一到 logs/engines/）
    setup_engine_logging(
        engine_id=_config.engine_id,
        log_level=_config.log_level,
        log_dir=_config.log_dir,
    )

    logger.info("===== VSR 视频字幕/水印移除引擎启动 =====")
    logger.info("配置: %s", _config.to_dict())

    # 初始化引擎
    _engine = VSRSubtitleRemoverEngine(_config)
    # v0.2.11+ 方案 B：lifespan startup 阶段异步刷新扫描（平台 API 已就绪）
    # __init__ 阶段的同步 _refresh_scan 只是 fallback，此处用平台 API 重新检测
    await _engine._refresh_scan_async()
    logger.info(
        "引擎实例化完成: state=%s, variants=%d, default_variant=%s",
        _engine.state_machine.state.value,
        len(_engine._available_variants),
        _engine.default_variant,
    )

    yield

    # 关闭清理：卸载模型
    logger.info("===== VSR 视频字幕/水印移除引擎关闭 =====")
    if _engine is not None and _engine.wrapper.is_loaded:
        try:
            await _engine._unload_impl()
        except Exception as e:
            logger.warning("关闭时卸载模型失败: %s", e)
    _engine = None


# ===== 创建 FastAPI 应用 =====

def create_app(config: EngineConfig | None = None) -> FastAPI:
    """创建 FastAPI 应用。

    Args:
        config: 引擎配置（None 则从命令行解析）

    Returns:
        FastAPI: 配置好的应用实例
    """
    global _config
    if config is not None:
        _config = config

    app = FastAPI(
        title="VSR Video Subtitle Remover Engine",
        description=(
            "AIReliable 平台的 VSR 视频字幕/水印移除引擎"
            "（PP-OCRv5 字幕检测 + 4 种 inpaint 算法：STTN-AUTO / STTN-DET / LAMA / ProPainter）"
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS（允许主控跨域调用，iframe 嵌入需要）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],                  # v0.1 简化，生产环境应限制
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ===== 注册 8 个标准端点 =====
    _register_endpoints(app)

    # ===== 挂载输出目录（/output/* 提供视频文件下载）=====
    if _config is not None:
        app.mount(
            "/output",
            StaticFiles(directory=str(_config.output_root)),
            name="output",
        )
        logger.info("输出目录挂载: /output -> %s", _config.output_root)

    # ===== 挂载原生 WebUI（webui/ 三件套）=====
    _mount_webui(app)

    return app


class FirstFrameRequest(BaseModel):
    """首帧提取请求（WebUI 涂抹选区用）。

    Attributes:
        video_path: 引擎侧视频绝对路径（/upload 返回值）
        frame_no: 要提取的帧号（0 起；首帧黑场时可换一帧）
    """

    video_path: str
    frame_no: int = 0


def _mount_webui(app: FastAPI) -> None:
    """挂载原生 WebUI 静态文件到 /ui（webui/ 三件套）。"""
    webui_dir = Path(__file__).parent / "webui"
    if webui_dir.is_dir() and (webui_dir / "index.html").exists():
        app.mount(
            "/ui",
            StaticFiles(directory=str(webui_dir), html=True),
            name="webui",
        )
        logger.info("原生 WebUI 挂载: /ui -> %s", webui_dir)
    else:
        logger.warning("WebUI 未挂载（webui/index.html 不存在）: %s", webui_dir)


def _register_endpoints(app: FastAPI) -> None:
    """注册 8 个标准 HTTP 端点 + 日志级别端点 + WebUI 专属端点。"""

    # 1. /health
    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """健康检查。"""
        return await get_engine().health()

    # 2. /capabilities
    @app.get("/capabilities")
    async def capabilities() -> dict:
        """能力声明（可用模型变体 / 支持的 inpaint 算法）。"""
        resp = await get_engine().capabilities()
        # 手动序列化 variants，确保 VSRVariantInfo 的扩展字段
        # （inpaint_mode / requires_ocr / max_resolution）被保留。
        # CapabilitiesResponse.variants 声明类型为 list[VariantInfo]，
        # Pydantic model_dump() 按声明类型序列化会丢弃子类扩展字段。
        result = resp.model_dump(exclude={"variants"})
        result["variants"] = [v.model_dump() for v in resp.variants]
        return result

    # 3. /load
    @app.post("/load")
    async def load(request: LoadRequest) -> dict:
        """校验模型文件 + 配置 inpaint 算法（VSR 惰性加载，快速返回）。

        请求体参考 LoadRequest schema：
            - variant: 模型变体名（VSR/STTN-AUTO / VSR/STTN-DET / VSR/LAMA / VSR/PROPAINTER）
        """
        try:
            response = await get_engine().load(request)
            return response.model_dump()
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.exception("加载失败")
            raise HTTPException(status_code=500, detail=str(e))

    # 4. /unload
    @app.post("/unload")
    async def unload() -> dict:
        """卸载模型（释放缓存 + 显存）。"""
        try:
            response = await get_engine().unload()
            return response.model_dump()
        except Exception as e:
            logger.exception("卸载失败")
            raise HTTPException(status_code=500, detail=str(e))

    # 5. /infer（SSE 流式）
    @app.post("/infer")
    async def infer(request: InferRequest):
        """视频字幕/水印移除推理（SSE 流式返回进度 + 最终视频 URL）。

        请求体参考 InferRequest schema：
            - variant: 模型变体名
            - inputs: VSRInputs 字段（video_path / sub_areas / ab_sections / detect_mode / params）
            - videos: 目标视频列表（videos[].path，优先于 inputs.video_path）
            - task_id: 可选任务 ID
        """
        engine = get_engine()

        # BaseEngine.infer 返回异步迭代器（SSE 帧）
        sse_stream = engine.infer(request)

        return StreamingResponse(
            sse_stream,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",          # Nginx 不缓冲
            },
        )

    # 6. /status
    @app.get("/status")
    async def status() -> dict:
        """引擎状态查询（含已加载变体 / 设备 / inpaint 模式）。"""
        return (await get_engine().status()).model_dump()

    # 7. /lora/load（v1 预留，不支持 LoRA）
    @app.post("/lora/load")
    async def lora_load(request: LoraLoadRequest) -> dict:
        """加载 LoRA（VSR v1 不支持，预留接口）。"""
        try:
            response = await get_engine().load_lora(request)
            return response.model_dump()
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("LoRA 加载失败")
            raise HTTPException(status_code=500, detail=str(e))

    # 8. /lora/unload（v1 预留，不支持 LoRA）
    @app.post("/lora/unload")
    async def lora_unload(request: LoraUnloadRequest) -> dict:
        """卸载 LoRA（VSR v1 不支持，预留接口）。"""
        try:
            response = await get_engine().unload_lora(request)
            return response.model_dump()
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("LoRA 卸载失败")
            raise HTTPException(status_code=500, detail=str(e))

    # ===== 通用 =====

    # 错误处理
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """HTTP 异常统一格式。"""
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(error={
                "type": exc.__class__.__name__,
                "message": str(exc.detail),
                "code": exc.status_code,
            }).model_dump(),
        )

    # v0.2.5+ T6.2：日志级别动态调整端点（GET /log_level + POST /log_level）
    register_log_level_endpoint(app)

    # ===== WebUI 专属端点 =====

    # 根路径 307 重定向到 /ui/
    @app.get("/", include_in_schema=False)
    async def _root_redirect() -> RedirectResponse:
        return RedirectResponse(url="/ui/", status_code=307)

    # 视频上传（WebUI）
    # 前端 FormData 上传 → 落盘 output/uploads/ → 返回引擎侧绝对路径
    # 供 /infer 的 inputs.video_path 引用
    @app.post("/upload")
    async def upload(file: UploadFile) -> dict:
        config = get_config()
        upload_dir = config.output_root / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        safe_name = Path(file.filename or "upload.mp4").name  # 防路径穿越
        dest = upload_dir / f"{uuid.uuid4().hex[:8]}_{safe_name}"  # uuid 前缀防重名
        size = 0
        try:
            with open(dest, "wb") as f:
                while True:
                    chunk = await file.read(1024 * 1024)  # 1MB 分块
                    if not chunk:
                        break
                    f.write(chunk)
                    size += len(chunk)
        finally:
            await file.close()
        logger.info("视频上传: %s (%d bytes) -> %s", safe_name, size, dest)
        return {
            "path": str(dest),
            "filename": safe_name,
            "size": size,
            # 静态挂载 /output 指向 output_root，uploads 子目录可直接经此播放
            "url": f"/output/uploads/{dest.name}",
        }

    # 首帧提取（WebUI 涂抹选区用）
    # 用 OpenCV 读取指定帧并编码为 JPEG base64 返回，与后端实际处理帧同源。
    # 不落盘文件（无清理负担），base64 体积 1080p 约 200-500KB，可接受。
    @app.post("/first_frame")
    async def first_frame(req: FirstFrameRequest) -> dict:
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


# ===== 入口 =====

# 模块级 app 实例（uvicorn 直接引用）
app: Optional[FastAPI] = None


def main() -> None:
    """命令行启动入口。"""
    global _config, app

    # 解析命令行参数
    _config = parse_args()

    # 配置日志（lifespan 也会配置，这里先配一遍让启动日志可见）
    setup_engine_logging(
        engine_id=_config.engine_id,
        log_level=_config.log_level,
        log_dir=_config.log_dir,
    )

    logger.info(
        "启动 VSR 视频字幕/水印移除引擎: host=%s, port=%d",
        _config.host, _config.port,
    )

    # 创建 app
    app = create_app(_config)

    # 用 uvicorn 启动
    import uvicorn
    uvicorn.run(
        app,
        host=_config.host,
        port=_config.port,
        log_level=_config.log_level.lower(),
    )


if __name__ == "__main__":
    main()
