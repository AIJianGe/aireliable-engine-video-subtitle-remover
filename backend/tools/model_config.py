import os
import logging

from backend.config import config, MODEL_BASE_DIR
from backend.tools.common_tools import merge_big_file_if_not_exists
from backend.tools.constant import SubtitleDetectMode

logger = logging.getLogger(__name__)

_MODEL_NAME_MAP = {
    SubtitleDetectMode.PP_OCRv5_MOBILE: "PP-OCRv5_mobile_det",
    SubtitleDetectMode.PP_OCRv5_SERVER: "PP-OCRv5_server_det",
}

# 模型布局（相对 MODEL_BASE_DIR，即 {AIRELIABLE_MODELS_ROOT}/{AIRELIABLE_MODEL_LAYOUT_BASE}）。
# 与 manifest.yaml model_files.target_dir 一一对应：
#   video/vsr/big-lama/        ← vsr-lama
#   video/vsr/sttn-auto/       ← vsr-sttn-auto
#   video/vsr/sttn-det/        ← vsr-sttn-det
#   video/vsr/propainter/      ← vsr-propainter
#   video/vsr/V5/ch_det/       ← vsr-ocr-server
#   video/vsr/V5/ch_det_fast/  ← vsr-ocr-mobile
class ModelConfig:
    def __init__(self):
        self.LAMA_MODEL_DIR = os.path.join(MODEL_BASE_DIR, 'big-lama')
        # STTN 权重为 safetensors 格式（Dev-Jahn 社区转换版，manifest 落地名 model.safetensors）
        self.STTN_AUTO_MODEL_PATH = os.path.join(MODEL_BASE_DIR, 'sttn-auto', 'model.safetensors')
        self.STTN_DET_MODEL_PATH = os.path.join(MODEL_BASE_DIR, 'sttn-det', 'model.safetensors')
        self.PROPAINTER_MODEL_DIR = os.path.join(MODEL_BASE_DIR, 'propainter')
        if config.subtitleDetectMode.value == SubtitleDetectMode.PP_OCRv5_MOBILE:
            self.DET_MODEL_DIR = os.path.join(MODEL_BASE_DIR, 'V5', 'ch_det_fast')
        elif config.subtitleDetectMode.value == SubtitleDetectMode.PP_OCRv5_SERVER:
            self.DET_MODEL_DIR = os.path.join(MODEL_BASE_DIR, 'V5', 'ch_det')
        else:
            raise ValueError(f"Invalid subtitle detect mode: {config.subtitleDetectMode.value}")
        self.DET_MODEL_NAME = _MODEL_NAME_MAP[config.subtitleDetectMode.value]

        # 平台下载器已落地「合并后的单文件」时 merge 为 no-op；
        # 目录不存在（模型未下载）时容错跳过，文件校验由 load() 兜底
        for model_dir, target_file in (
            (self.LAMA_MODEL_DIR, 'big-lama.pt'),
            (self.PROPAINTER_MODEL_DIR, 'ProPainter.pth'),
        ):
            if os.path.isdir(model_dir):
                try:
                    merge_big_file_if_not_exists(model_dir, target_file)
                except Exception as e:
                    logger.warning("模型分片合并跳过（%s）: %s", model_dir, e)
