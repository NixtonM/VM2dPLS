import argparse
import coloredlogs
import logging

from VM2dPLS.io import loader

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument('filepath', type=str, help='Path to profile file')
    parser.add_argument('-s', '--scanner_type', type=str, choices=['ZF5016', 'other'], required=True,
                        help='Scanner type')
    parser.add_argument('-f', '--file_type', type=str, choices=['ascii', 'bin'], required=True,
                        help='Ascii or binary file type')
    parser.add_argument('-fm', '--filepath_meta', type=str, metavar='META_FILE', help='Path to meta file')
    parser.add_argument('-t', '--time_window', type=int, nargs=2, metavar=('START', 'END'), default=None,
                        help='Time window of interest in seconds')
    parser.add_argument('-c', '--cache_interim', action='store_true',
                        help='Save intermediate results as binaries (pickle) for quick restarts')
    parser.add_argument('-r', '--restart_from_cache', action='store_true',
                        help='Use last intermediate results (with same parameters) as starting point')
    parser.add_argument('-v', '--verbose', action='store_true', help='Set logging output to DEBUG level')
    args = parser.parse_args()

    logger = logging.getLogger('VM2dPLS')

    if args.verbose:
        _logging_level = logging.DEBUG
    else:
        _logging_level = logging.INFO

    coloredlogs.install(level=_logging_level, logger=logger, milliseconds=True,
                        fmt='%(asctime)s %(hostname)s %(name)s[%(process)d] %(levelname)s — %(funcName)s:%(lineno)d - %(message)s')

    logging.info('Let\'s get started')
    logging.debug('Debug message')
    loader(args)
