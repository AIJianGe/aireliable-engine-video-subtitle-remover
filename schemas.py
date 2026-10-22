"""VSR 视频字幕/水印移除引擎 Pydantic 数据模型。

扩展 SDK 的 VariantInfo / InferRequest.inputs，定义视频去字幕专属字段。

variant = inpaint 算法（4 个）：
    - VSR/STTN-AUTO   ：智能擦除（默认，无需 OCR，按选区直接生成 mask）
    - VSR/STTN-DET    ：检测式擦除（需 OCR 定位字幕）
    - VSR/LAMA        ：图像级 inpainting（需 OCR）
    - VSR/PROPAINTER  ：视频级 inpainting（需 OCR，720p 需 ~25G 显存）

任务类型（task_type）：
    - video_subtitle_remover：视频硬字幕/水印移除（保留原音频）

infer 入参（放入 InferRequest.inputs，均由 WebUI/网关透传）：
    - video_path  ：目标视频绝对路径（videos[].path 或 inputs.video_path）
    - sub_areas   ：字幕区域列表 [[ymin,ymax,xmin,xmax], ...]
                    （空数组 = 全屏区域；STTN-AUTO 必填，其余算法可自动检测）
    - ab_sections ：可选的 AB 段时间区间 [[start,end], ...]（None = 处理全部帧）
    - detect_mode ：OCR 检测模式 PP_OCRv5_SERVER / PP_OCRv5_MOBILE
                    （STTN-DET / LAMA / PROPAINTER 生效）
    - params      ：算法参数透传（可选，覆盖 config 默认值）
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from aireliable_engine import VariantInfo

# ===== 能力声明 =====
CAPABILITY = "video-editing"           # 能力类别（与 manifest 设计一致）
ENGINE_TYPE = "video"

# ===== 任务类型 =====
TASK_TYPE = "video_subtitle_remover"

# ===== inpaint 算法与 OCR 检测模式枚举值（与 backend/tools/constant.py 一致）=====
INPAINT_MODES = ("sttn-auto", "sttn-det", "lama", "propainter", "opencv")
DETECT_MODES = ("PP_OCRv5_SERVER", "PP_OCRv5_MOBILE")


class VSRVariantInfo(VariantInfo):
    """VSR 变体能力声明（扩展 SDK VariantInfo）。

    在标准字段之外补充视频去字幕特有信息：
        - inpaint_mode：inpaint 算法（sttn-auto / sttn-det / lama / propainter）
        - requires_ocr：是否需要 OCR 检测能力
    """

    model_config = ConfigDict(extra="allow")

    inpaint_mode: str = Field(
        "sttn-auto",
        description="inpaint 算法：sttn-auto / sttn-det / lama / propainter",
    )
    requires_ocr: bool = Field(
        False,
        description="该算法是否需要 OCR 字幕检测（STTN-DET / LAMA / PROPAINTER 需要）",
    )
    max_resolution: str = Field(
        "1920x1080",
        description="建议最大输入分辨率（PROPAINTER 720p 需 ~25G 显存）",
    )


class VSRInputs(BaseModel):
    """视频字幕移除推理输入参数（放入 InferRequest.inputs）。

    所有字段均有默认值，WebUI / 网关按需覆盖。
    """

    model_config = ConfigDict(extra="allow")

    # ===== 目标视频 =====
    # 优先使用 InferRequest.videos[].path，此处为兜底/直连场景
    video_path: Optional[str] = Field(
        None, description="目标视频绝对路径（直连链路用；网关链路走 videos[].path）"
    )

    # ===== 字幕区域 =====
    # 归一化坐标 [ymin, ymax, xmin, xmax]（0-1 比例）
    # 空数组 = 全屏区域（与上游 GUI 的"全屏处理"语义一致）
    sub_areas: list[list[float]] = Field(
        default_factory=list,
        description="字幕区域列表（归一化 [ymin,ymax,xmin,xmax]，空=全屏；"
        "STTN-AUTO 必填，其余算法为空时走 OCR 自动检测）",
    )

    # ===== AB 段时间区间 =====
    ab_sections: Optional[list[list[int]]] = Field(
        None,
        description="处理时间段区间 [[start_frame, end_frame], ...]（None=全部帧）",
    )

    # ===== OCR 检测模式（STTN-DET / LAMA / PROPAINTER 生效）=====
    detect_mode: str = Field(
        "PP_OCRv5_SERVER",
        description=f"OCR 字幕检测模式：{' / '.join(DETECT_MODES)}",
    )

    # ===== 算法参数透传（可选，覆盖 config 默认值）=====
    # 键名与 backend/config.py 的 _ConfigValue 属性名一致（如
    # subtitleYXAxisDifferencePixel / sttnNeighborStride / propainterMaxLoadNum）
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="算法参数透传（可选，覆盖引擎配置默认值）",
    )


def normalize_sub_areas(sub_areas: list[list[float]]) -> list[tuple[int, int, int, int]]:
    """把归一化 [ymin,ymax,xmin,xmax] 选区转换为像素坐标（上游格式）。

    上游 SubtitleRemover.sub_areas 的元素是像素坐标元组 (ymin, ymax, xmin, xmax)。
    本函数根据视频帧尺寸换算：空数组返回空（由 SubtitleRemover.run 兜底为全屏）。

    Args:
        sub_areas: 归一化选区列表（0-1 比例）

    Returns:
        像素坐标元组列表 [(ymin, ymax, xmin, xmax), ...]
    """
    return []  # 像素换算需要帧尺寸，由 vsr_wrapper 在拿到视频尺寸后调用 _to_pixel
