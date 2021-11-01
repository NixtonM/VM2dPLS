import logging
import time
from typing import Dict, Any
import hashlib
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
