import logging
import time
from typing import Dict, Any
import hashlib
import numpy as np
import json


def dict_hash(dictionary: Dict[str, Any]) -> str:
    """MD5 hash of a dictionary."""
    dhash = hashlib.md5()
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()


def timeit(func):
    """
    Decorator for measuring function's running time.
    """
    def measure_time(*args, **kw):
        logger = logging.getLogger(__name__)
        start_time = time.time()
        logger.debug(f"Starting {func.__qualname__} ")
        result = func(*args, **kw)
        run_time = time.time() - start_time
        logger.debug(f"Processing time of {func.__qualname__}: {run_time} seconds.")
        return result

    return measure_time


def reject_outliers(data, m=2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return s<m


def convert_2d_polar_to_xy(distance, v_angle):
    # v_angle in nadir direction, clockwise in rad
    alpha = np.float64(np.mod(3/2*np.pi - v_angle, 2*np.pi))
    x = distance * np.cos(alpha)
    y = distance * np.sin(alpha)
    return np.array([x, y])


def convert_xy_to_2d_polar(x, y):
    # v_angle in nadir direction, clockwise in rad
    distance = np.hypot(x, y)
    v_angle = np.arctan2(-x, -y)
    return np.array([distance, v_angle])

