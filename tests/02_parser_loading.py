import argparse
import coloredlogs
import logging
from datetime import timedelta

from VM2dPLS.io import ScanHandlerZF5016
from VM2dPLS.core import cut_to_vertical_angle_range, select_vertical_angle_range_GUI, plot_profile, animate_profiles, \
    regional_binning, fft_profile_point

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument('filepath', type=str, help='Path to profile file')
    parser.add_argument('-s', '--scanner_type', type=str, choices=['ZF5016', 'other'], required=True,
                        help='Scanner type')
    parser.add_argument('-f', '--file_type', type=str, choices=['ascii', 'bin'], required=True,
                        help='Ascii or binary file type')
    parser.add_argument('-fm', '--filepath_meta', type=str, metavar='META_FILE', help='Path to meta file', default=None)
    parser.add_argument('-t', '--time_window', type=int, nargs=2, metavar=('START', 'END'), default=None,
                        help='Time window of interest in seconds')
    parser.add_argument('-c', '--cache_interim', action='store_true',
                        help='Save intermediate results as binaries (pickle) for quick restarts')
    parser.add_argument('-r', '--restart_from_cache', action='store_true',
                        help='Use last intermediate results (with same parameters) as starting point')
    parser.add_argument('-v', '--verbose', action='store_true', help='Set logging output to DEBUG level')
    args = parser.parse_args()

    logger = logging.getLogger(__name__)

    if args.verbose:
        _logging_level = logging.DEBUG
    else:
        _logging_level = logging.INFO

    coloredlogs.install(level=_logging_level, milliseconds=True,
                        fmt='%(asctime)s %(hostname)s %(name)s[%(process)d] %(levelname)s — %(funcName)s:%(lineno)d - %(message)s')

    logger.info('Let\'s get started')
    logger.debug('Debug message')

    if args.scanner_type == 'ZF5016':
        # profile_handler = ProfileHandlerZF5016(args.filepath_meta,
        #                                        tuple(timedelta(seconds=t) for t in args.time_window),
        #                                        measurement_file_path=args.filepath, cache_interim=args.cache_interim,
        #                                        restart_from_cache=args.restart_from_cache)
        scan_handler = ScanHandlerZF5016(vars(args))

    last_cached_step = scan_handler.load_data(args.file_type == 'ascii')

    if last_cached_step == 0:
        theta_borders = select_vertical_angle_range_GUI(scan_handler)
        cut_to_vertical_angle_range(scan_handler, *theta_borders)
        plot_profile(scan_handler)
        animate_profiles(scan_handler)
        regional_binning(scan_handler)
        last_cached_step = 1
    if last_cached_step == 1:
        # plot_profile(scan_handler)
        #
        fft_profile_point(scan_handler, profile_point=30)
        # for i in range(list(scan_handler.profiles.data.values())[0].data.shape[0]):
        #     fft_profile_point(scan_handler, profile_point=i)


    print(1)

