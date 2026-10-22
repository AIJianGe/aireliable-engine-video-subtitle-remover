"""VSR 视频字幕/水印移除引擎"""
__version__ = "0.1.0"

# 强制离线模式：必须在引擎任何模块导入 huggingface_hub / paddlex 之前设置
# VSR 模型权重已下载到本地（平台模型管理），禁止运行时自动联网下载
import os as _os
_os.environ["HF_HUB_OFFLINE"] = "1"
_os.environ["TRANSFORMERS_OFFLINE"] = "1"
_os.environ["HF_HUB_DISABLE_XET"] = "1"
_os.environ["HF_DATASETS_OFFLINE"] = "1"

# huggingface_hub.constants 在首次导入时读取环境变量并缓存为模块属性
# 仅设 os.environ 不够（constants.HF_HUB_OFFLINE 可能已是 False），必须 patch 模块属性
try:
    import huggingface_hub.constants as _hf_constants
    _hf_constants.HF_HUB_OFFLINE = True
    _hf_constants.HF_HUB_DISABLE_IMPLICIT_TOKEN = True
    # ENDPOINT 置空避免任何对 huggingface.co 的请求
    _hf_constants.ENDPOINT = ""
except ImportError:
    pass
