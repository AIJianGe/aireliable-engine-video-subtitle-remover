import os
import sys
import ctypes
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

video_extensions = {
    '.mp4', '.m4a', '.m4v', '.f4v', '.f4a', '.m4b', '.m4r', '.f4b', '.mov',
    '.3gp', '.3gp2', '.3g2', '.3gpp', '.3gpp2', '.ogg', '.oga', '.ogv', '.ogx',
    '.wmv', '.wma', '.asf', '.webm', '.flv', '.avi', '.gifv', '.mkv', '.rm',
    '.rmvb', '.vob', '.dvd', '.mpg', '.mpeg', '.mp2', '.mpe', '.mpv', '.mpg',
    '.mpeg', '.m2v', '.svi', '.3gp', '.mxf', '.roq', '.nsv', '.flv', '.f4v',
    '.f4p', '.f4a', '.f4b'
}

image_extensions = {
    '.jpg', '.jpeg', '.jpe', '.jif', '.jfif', '.jfi', '.png', '.gif',
    '.webp', '.tiff', '.tif', '.psd', '.raw', '.arw', '.cr2', '.nrw',
    '.k25', '.bmp', '.dib', '.heif', '.heic', '.ind', '.indd', '.indt',
    '.jp2', '.j2k', '.jpf', '.jpx', '.jpm', '.mj2', '.svg', '.svgz',
    '.ai', '.eps', '.ico'
}


def is_video_file(filename):
    return os.path.splitext(filename)[-1].lower() in video_extensions


def is_image_file(filename):
    return os.path.splitext(filename)[-1].lower() in image_extensions


def is_video_or_image(filename):
    file_extension = os.path.splitext(filename)[-1].lower()
    # 检查扩展名是否在定义的视频或图片文件后缀集合中
    return file_extension in video_extensions or file_extension in image_extensions

def merge_big_file_if_not_exists(dir, file, man_filename = None):
    """分片文件合并（替代原 fsplit 依赖）。

    平台下载器默认按单文件落地，此函数仅在目录中存在 fsplit 风格的分片
    （fs_manifest.csv + fs_xxx 分片）时才执行合并；目标文件已存在且
    非空时为 no-op。
    """
    target_path = os.path.join(dir, file)
    if not os.path.exists(target_path) or os.path.getsize(target_path) == 0:
        _merge_fs_chunks(dir, file, man_filename)


def _merge_fs_chunks(dir, file, man_filename=None):
    """合并 fsplit 风格分片到目标文件。

    fsplit 生成的清单（默认 fs_manifest.csv）每行一列分片文件名（CSV 首列），
    按行序拼接所有分片内容到目标文件。无清单或清单缺失时抛错（由调用方容错）。
    """
    manifest_name = man_filename if man_filename else 'fs_manifest.csv'
    manifest_path = os.path.join(dir, manifest_name)
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            f"分片清单不存在：{manifest_path}（目标文件 {file} 未合并，请重新下载）"
        )

    target_path = os.path.join(dir, file)
    total_written = 0
    with open(target_path, 'wb') as out:
        with open(manifest_path, 'r', encoding='utf-8-sig', errors='replace') as mf:
            for line in mf:
                line = line.strip()
                if not line:
                    continue
                chunk_name = line.split(',')[0]
                # 跳过 fsplit 清单表头行（filename,filesize,encoding,header）
                if chunk_name == 'filename':
                    continue
                chunk_path = os.path.join(dir, chunk_name)
                if not os.path.isfile(chunk_path):
                    raise FileNotFoundError(
                        f"缺少分片文件：{chunk_path}（目标文件 {file} 合并失败，请重新下载）"
                    )
                with open(chunk_path, 'rb') as cf:
                    while True:
                        data = cf.read(1024 * 1024)
                        if not data:
                            break
                        out.write(data)
                        total_written += len(data)
    logger.info("分片合并完成：%s → %s（%d 字节）", dir, file, total_written)
    if total_written == 0:
        raise RuntimeError(f"分片合并结果为空文件：{target_path}")

def get_readable_path(path):
    if sys.platform != 'win32':
        return path
    buf = ctypes.create_unicode_buffer(4096)
    ctypes.windll.kernel32.GetShortPathNameW(path, buf, 4096)
    return buf.value

def read_image(path):
    if os.path.getsize(path) > 100*1024*1024: # 100MB
        print(f"Image {path} is too large, skip")
        return None
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    if img is not None and img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img