"""VSR 后端轻量化配置（服务化版，剥离 qfluentwidgets 桌面 GUI 依赖）。

上游 video-subtitle-remover 的 backend/config.py 直接 import qfluentwidgets
（PySide6 GUI 库），服务化 import 会报错。此文件以「普通属性 + .value 包装」
等价替换上游 QConfig 体系，保持访问方式一致：

    - 上游：config.inpaintMode.value → OptionsConfigItem
    - 本版：config.inpaintMode.value → _ConfigValue(枚举)

对外提供的全局对象（与上游同名，backend/ 其余代码零改动）：
    - VERSION：项目版本号
    - config：全局配置单例
    - tr     ：翻译表（服务化固定加载 en.ini）
    - BASE_DIR：本文件所在目录（backend/），供 ffmpeg 等相对路径使用
    - MODEL_BASE_DIR：平台模型根目录下的 VSR 布局（AIRELIABLE_MODEL_LAYOUT_BASE）

模型目录重定向（平台规范）：
    上游模型路径为 {BASE_DIR}/models/<name>；平台下模型由平台下载器
    下载到 {AIRELIABLE_MODELS_ROOT}/{AIRELIABLE_MODEL_LAYOUT_BASE}/<name>。
    ModelConfig 据此读取 MODEL_BASE_DIR 构造路径，与 manifest.yaml 的
    model_layout.base 保持一致。
"""

import os
import configparser
from pathlib import Path

from backend.tools.constant import InpaintMode, SubtitleDetectMode

# ===== 项目元信息 =====
VERSION = "1.4.0"
PROJECT_HOME_URL = "https://github.com/YaoFANGUK/video-subtitle-remover"

# 硬件加速选项开关
HARDWARD_ACCELERATION_OPTION = True

# ===== 模型目录重定向（平台注入）=====
# 优先级：AIRELIABLE_MODEL_LAYOUT_BASE（v0.2.1 起由 EngineManager 注入，
# 值为 manifest.model_layout.base，如 "video/vsr/"）> 硬编码默认
def _resolve_model_base_dir() -> str:
    env_val = os.environ.get("AIRELIABLE_MODEL_LAYOUT_BASE")
    if env_val:
        return env_val.rstrip("/\\") + os.sep
    # 硬编码默认（独立调试场景）：与 manifest model_layout.base 保持一致
    return "video/vsr" + os.sep


# 本文件所在目录（backend/）
BASE_DIR = str(Path(os.path.abspath(__file__)).parent)
# 模型布局根：{AIRELIABLE_MODELS_ROOT}/{layout}，无 env 时回退 backend/models/
_MODELS_ROOT = os.environ.get("AIRELIABLE_MODELS_ROOT", os.path.join(BASE_DIR, "..", "models"))
MODEL_BASE_DIR = os.path.abspath(os.path.join(_MODELS_ROOT, _resolve_model_base_dir()))

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# ===== 轻量配置值包装 =====
class _ConfigValue:
    """与上游 ConfigItem 的 .value 访问兼容的包装。

    支持运行时更新（engine.infer 透传参数时调用 config.xxx.value = 新值）。
    """

    def __init__(self, value):
        self.value = value


class Config:
    """VSR 全局配置（服务化版）。

    仅保留 backend/ 业务代码实际读取的参数项（GUI 相关的窗口位置/语言/
    检查更新等已剥离）。所有参数可用环境变量覆盖（供平台注入默认值）。
    """

    def __init__(self):
        # 界面语言（服务化固定 en，翻译表加载用）
        self.interface = _ConfigValue("en")

        # 【选区】默认值格式 "ymin,ymax,xmin,xmax;..."，服务化由 infer 请求传入
        self.subtitleSelectionAreas = _ConfigValue("0.88,0.99,0.15,0.85")

        # 【inpaint 算法】
        self.inpaintMode = _ConfigValue(
            self._env_enum("VSR_INPAINT_MODE", InpaintMode, InpaintMode.STTN_AUTO)
        )
        # 【字幕检测模式】
        self.subtitleDetectMode = _ConfigValue(
            self._env_enum("VSR_DETECT_MODE", SubtitleDetectMode, SubtitleDetectMode.PP_OCRv5_SERVER)
        )

        # 【像素点偏差】字幕检测参数
        self.subtitleYXAxisDifferencePixel = _ConfigValue(self._env_int("VSR_YX_AXIS_DIFF", 10))
        self.subtitleAreaDeviationPixel = _ConfigValue(self._env_int("VSR_AREA_DEVIATION", 25))
        self.subtitleAreaYAxisDifferencePixel = _ConfigValue(self._env_int("VSR_AREA_Y_DIFF", 20))
        self.subtitleAreaPixelToleranceYPixel = _ConfigValue(self._env_int("VSR_TOLERANCE_Y", 20))
        self.subtitleAreaPixelToleranceXPixel = _ConfigValue(self._env_int("VSR_TOLERANCE_X", 20))
        self.subtitleTimelineBackwardFrameCount = _ConfigValue(self._env_int("VSR_TIMELINE_BACKWARD", 3))
        self.subtitleTimelineForwardFrameCount = _ConfigValue(self._env_int("VSR_TIMELINE_FORWARD", 3))

        # 【STTN 算法参数】
        self.sttnNeighborStride = _ConfigValue(self._env_int("VSR_STTN_NEIGHBOR_STRIDE", 5))
        self.sttnReferenceLength = _ConfigValue(self._env_int("VSR_STTN_REFERENCE_LENGTH", 10))
        self.sttnMaxLoadNum = _ConfigValue(self._env_int("VSR_STTN_MAX_LOAD_NUM", 50))

        # 【ProPainter 算法参数】
        self.propainterMaxLoadNum = _ConfigValue(self._env_int("VSR_PROPAINTER_MAX_LOAD_NUM", 70))

        # 【硬件加速】（onnxruntime 硬件加速开关）
        self.hardwareAcceleration = _ConfigValue(self._env_bool("VSR_HARDWARE_ACCEL", HARDWARD_ACCELERATION_OPTION))

        # 【保存目录】（服务化固定为空 = 引擎输出目录）
        self.saveDirectory = _ConfigValue("")

    @staticmethod
    def _env_enum(env_name: str, enum_cls, default):
        raw = os.environ.get(env_name)
        if raw is None:
            return default
        for member in enum_cls:
            if raw == member.value or raw == member.name:
                return member
        return default

    @staticmethod
    def _env_int(env_name: str, default: int) -> int:
        raw = os.environ.get(env_name)
        try:
            return int(raw) if raw is not None else default
        except ValueError:
            return default

    @staticmethod
    def _env_bool(env_name: str, default: bool) -> bool:
        raw = os.environ.get(env_name)
        if raw is None:
            return default
        return raw.strip().lower() in ("1", "true", "yes", "on")

    def getSttnMaxLoadNum(self):
        """与上游一致：STTN 最大同时处理帧数（不小于 步长×参考帧数）。"""
        return max(self.sttnMaxLoadNum.value,
                   self.sttnNeighborStride.value * self.sttnReferenceLength.value)

    def set(self, item: _ConfigValue, value):
        """兼容上游 config.set(ConfigItem, value)（迁移逻辑等场景）。"""
        item.value = value


# ===== 全局配置单例 =====
config = Config()

# 服务化固定使用 en 翻译表（backend 业务代码通过 tr['Main'][...] 取文案）
TRANSLATION_FILE = os.path.join(BASE_DIR, 'interface', f"{config.interface.value}.ini")
tr = configparser.ConfigParser()
tr.read(TRANSLATION_FILE, encoding='utf-8')
