# src/compat.py
# ============================================================
# Compatibility layer cho các thay đổi API giữa các phiên bản aeon
#
# Breaking changes liên quan:
#   aeon < 0.10 : MiniRocket(num_kernels=..., random_state=...)
#   aeon >= 0.10: MiniRocket(n_kernels=...,   random_state=...)
#   aeon >= 0.11: random_state có thể bị đổi thành seed
# ============================================================

import inspect
from aeon.transformations.collection.convolution_based import MiniRocket


def _get_minirocket_api():
    """
    Kiểm tra tên tham số thực tế trong phiên bản aeon đang dùng.
    Trả về dict ánh xạ: {'kernels_param': 'n_kernels' hoặc 'num_kernels'}
    """
    sig    = inspect.signature(MiniRocket.__init__)
    params = list(sig.parameters.keys())

    if "n_kernels" in params:
        kernels_param = "n_kernels"
    elif "num_kernels" in params:
        kernels_param = "num_kernels"
    else:
        kernels_param = None  # Dùng default, không truyền tham số

    has_random_state = "random_state" in params

    return {
        "kernels_param":    kernels_param,
        "has_random_state": has_random_state,
        "all_params":       params,
    }


# Cache kết quả detect để không gọi lại nhiều lần
_AEON_API = None

def _get_cached_api():
    global _AEON_API
    if _AEON_API is None:
        _AEON_API = _get_minirocket_api()
        print(f"[compat] aeon MiniRocket API detected: "
              f"kernels_param='{_AEON_API['kernels_param']}', "
              f"has_random_state={_AEON_API['has_random_state']}")
    return _AEON_API


def make_minirocket(num_kernels: int = 10_000, random_state: int = 42) -> MiniRocket:
    """
    Factory function tạo MiniRocket tương thích với mọi phiên bản aeon.

    Parameters
    ----------
    num_kernels : int
        Số kernel (mặc định 10,000)
    random_state : int
        Seed cho reproducibility

    Returns
    -------
    MiniRocket instance đã được cấu hình đúng
    """
    api    = _get_cached_api()
    kwargs = {}

    # Tham số số kernel
    if api["kernels_param"] is not None:
        kwargs[api["kernels_param"]] = num_kernels

    # Tham số random state
    if api["has_random_state"]:
        kwargs["random_state"] = random_state

    return MiniRocket(**kwargs)


def get_aeon_version() -> str:
    """Trả về version string của aeon."""
    try:
        import aeon
        return aeon.__version__
    except Exception:
        return "unknown"
