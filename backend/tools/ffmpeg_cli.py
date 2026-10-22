import logging
import os
import stat

import platform
from .common_tools import merge_big_file_if_not_exists
from backend.config import BASE_DIR

logger = logging.getLogger(__name__)

class FFmpegCLI:

    """
    进程管理器类，用于管理子进程的生命周期
    使用弱引用避免内存泄漏
    """
    _instance = None

    @classmethod
    def instance(cls):
        """单例模式获取实例"""
        if cls._instance is None:
            cls._instance = FFmpegCLI()
        return cls._instance

    def __init__(self):
        # 平台注入 AIRELIABLE_FFMPEG_PATH 时优先使用外部 ffmpeg（如平台自带）
        ext_ffmpeg = os.environ.get("AIRELIABLE_FFMPEG_PATH")
        self._external_ffmpeg = ext_ffmpeg if ext_ffmpeg else ""
        if not self._external_ffmpeg and os.path.exists(self.ffmpeg_path):
            try:
                os.chmod(self.ffmpeg_path, stat.S_IRWXU + stat.S_IRWXG + stat.S_IRWXO)
            except OSError:
                pass

    @property
    def ffmpeg_path(self):
        if self._external_ffmpeg:
            return self._external_ffmpeg
        system = platform.system()
        if system == "Windows":
            ffmpeg_dir = os.path.join(BASE_DIR, 'ffmpeg', 'win_x64')
            if os.path.isdir(ffmpeg_dir):
                try:
                    merge_big_file_if_not_exists(ffmpeg_dir, 'ffmpeg.exe')
                except Exception as e:
                    logger.warning("ffmpeg 分片合并跳过: %s", e)
            return os.path.join(ffmpeg_dir, 'ffmpeg.exe')
        elif system == "Linux":
            return os.path.join(BASE_DIR, 'ffmpeg', 'linux_x64', 'ffmpeg')
        else:
            return os.path.join(BASE_DIR, 'ffmpeg', 'macos', 'ffmpeg')