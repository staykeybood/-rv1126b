"""
简单图像增强。
弱光、逆光时可以在配置里打开 enhance；它会做自动 gamma 和 CLAHE。
板端算力紧张时建议默认关闭，只有画面质量明显影响检测时再开。
"""

from __future__ import annotations

import cv2
import numpy as np


def enhance_frame(
    frame: np.ndarray,
    enable_clahe: bool = True,
    enable_gamma: bool = True,
    enable_denoise: bool = False,
    clahe_clip: float = 2.0,
    denoise_h: float = 4.0,
) -> np.ndarray:
    """增强弱光/逆光画面，不改变图像尺寸。"""
    out = frame
    if enable_gamma:
        out = auto_gamma(out)
    if enable_clahe:
        out = clahe_luma(out, clip_limit=clahe_clip)
    if enable_denoise:
        out = cv2.fastNlMeansDenoisingColored(out, None, denoise_h, denoise_h, 7, 21)
    return out


def auto_gamma(frame: np.ndarray, target_luma: float = 115.0) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean = float(np.mean(gray))
    if mean <= 1.0:
        return frame

    gamma = np.log(max(target_luma, 1.0) / 255.0) / np.log(mean / 255.0)
    gamma = float(np.clip(gamma, 0.55, 1.75))
    table = ((np.arange(256, dtype=np.float32) / 255.0) ** gamma * 255.0).clip(0, 255)
    return cv2.LUT(frame, table.astype(np.uint8))


def clahe_luma(frame: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_chan = clahe.apply(l_chan)
    merged = cv2.merge((l_chan, a_chan, b_chan))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
