import logging
import time


def timeit(func):
    """
    Decorator for measuring function's running time.
    """
    def measure_time(*args, **kw):
        logger = logging.getLogger('VM2dPLS')
        start_time = time.time()
        result = func(*args, **kw)
        run_time = time.time() - start_time
        logger.debug(f"Processing time of {func.__qualname__}: {run_time} seconds.")
        return result

    return measure_time
