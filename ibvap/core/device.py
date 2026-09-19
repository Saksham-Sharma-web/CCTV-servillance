"""
Centralized Hardware & Device Management Subsystem.
Provides automatic detection and allocation of NVIDIA CUDA GPU acceleration
when available, with seamless, zero-error fallback to CPU execution.
Guarantees:
1. Automatic startup hardware detection (CUDA vs CPU).
2. Zero external GPU or driver dependencies when running on CPU-only machines.
3. Centralized PyTorch, OpenCV DNN, and Paddle device routing.
4. Logging of actual runtime devices used by each model subsystem.
5. CPU thread pool protection against library thread-starvation.
"""

from typing import Tuple, Optional, Dict, Any
import os
import logging
import torch
import cv2

logger = logging.getLogger("ibvap.core.device")


def get_torch_device(device_str: Optional[str] = "auto") -> torch.device:
    """
    Resolves the active PyTorch device.
    If device_str is 'auto' (or None):
      - Returns torch.device('cuda:0') if CUDA is available.
      - Returns torch.device('cpu') otherwise.
    If an explicit device is requested (e.g. 'cuda', 'cuda:0', 'cpu'):
      - Validates availability before assignment, falling back to 'cpu' if unavailable.
    """
    if device_str is None or device_str.lower() == "auto":
        if torch.cuda.is_available():
            dev = torch.device("cuda:0")
            logger.debug(f"[Device] Auto-selected CUDA GPU: {torch.cuda.get_device_name(0)}")
            return dev
        return torch.device("cpu")

    target = device_str.strip().lower()
    if target.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(target)
        logger.warning(f"[Device] Requested '{target}' but CUDA is not available. Falling back to CPU.")
        return torch.device("cpu")

    return torch.device("cpu")


def get_opencv_dnn_target() -> Tuple[int, int]:
    """
    Resolves OpenCV DNN backend and target.
    Returns (DNN_BACKEND_CUDA, DNN_TARGET_CUDA) if OpenCV has CUDA support and GPU is present.
    Otherwise returns (DNN_BACKEND_OPENCV, DNN_TARGET_CPU).
    """
    try:
        if hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            return cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA
    except Exception as e:
        logger.debug(f"[Device] OpenCV CUDA check returned: {e}")

    return cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU


def get_paddle_device() -> str:
    """
    Resolves PaddleOCR execution device string ('gpu:0' vs 'cpu').
    Only returns GPU if Paddle was compiled with CUDA and GPU is accessible.
    """
    try:
        import paddle
        if paddle.is_compiled_with_cuda() and torch.cuda.is_available():
            return "gpu:0"
    except Exception:
        pass
    return "cpu"


def ensure_cpu_thread_health(target_threads: Optional[int] = None) -> int:
    """
    Ensures PyTorch maintains optimal CPU thread count (e.g. 6) and does not
    get starved by third-party libraries (like PaddleX) that force torch.set_num_threads(1).
    """
    if torch.cuda.is_available():
        return torch.get_num_threads()

    if target_threads is None:
        cpu_count = os.cpu_count() or 4
        target_threads = min(6, cpu_count)

    curr = torch.get_num_threads()
    if curr < target_threads:
        torch.set_num_threads(target_threads)
        logger.debug(f"[Device] Restored PyTorch CPU threads: {curr} -> {target_threads}")
    return torch.get_num_threads()


def log_device_summary() -> Dict[str, Any]:
    """
    Logs and returns a summary of the active devices across all AI subsystems.
    """
    torch_dev = get_torch_device()
    cv_backend, cv_target = get_opencv_dnn_target()
    paddle_dev = get_paddle_device()

    summary = {
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "pytorch_device": str(torch_dev),
        "opencv_dnn_backend": "CUDA" if cv_backend == cv2.dnn.DNN_BACKEND_CUDA else "OPENCV_CPU",
        "paddle_device": paddle_dev,
        "torch_threads": torch.get_num_threads(),
    }

    logger.info("=" * 60)
    logger.info("IBVAP HARDWARE & ACCELERATION CONFIGURATION")
    logger.info(f"  CUDA Available:       {summary['cuda_available']}")
    logger.info(f"  Active GPU:           {summary['gpu_name']}")
    logger.info(f"  PyTorch Models:       {summary['pytorch_device']}")
    logger.info(f"  OpenCV YuNet Backend: {summary['opencv_dnn_backend']}")
    logger.info(f"  PaddleOCR Engine:     {summary['paddle_device']}")
    logger.info(f"  PyTorch CPU Threads:  {summary['torch_threads']}")
    logger.info("=" * 60)

    return summary
