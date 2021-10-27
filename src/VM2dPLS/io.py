from datetime import datetime, timedelta
from pathlib import Path
import re
from typing import Union, Tuple
from VM2dPLS.data import MeasurementGroupZF5016, ProfileMeasurementGroup
from VM2dPLS.utility import timeit


def loader(args):
    if args.time_window:
        args.time_window = tuple(timedelta(seconds=t) for t in args.time_window)
    if args.scanner_type == 'ZF5016' and args.file_type == 'ascii':
        profiles = load_ZF_data_from_ascii(args.filepath, args.filepath_meta, args.time_window)

    print(1)

@timeit
def load_ZF_data_from_ascii(measurement_file_path: Union[str, Path], meta_file_path: Union[str, Path],
                            time_region_of_interest: Union[None, Tuple[datetime, datetime],
                                                           Tuple[timedelta, timedelta]] = None) -> 'ProfileMeasurementGroup':
    measurement_file_path = Path(measurement_file_path)
    meta_file_path = Path(meta_file_path)

    start_time, end_time = load_ZF_meta_from_ascii(meta_file_path)

    measurements = MeasurementGroupZF5016.from_ZF_ascii_file(measurement_file_path)

    measurements.profile_and_point_id_calculation_ZF()
    measurements.timestamp_interpolation(start_time, end_time)

    if time_region_of_interest:
        if isinstance(time_region_of_interest[0], timedelta):
            time_region_of_interest = tuple(start_time + t for t in time_region_of_interest)
        measurements.time_filter(*time_region_of_interest)

    # TODO: Check if necessary..might cause runtime to explode unnecessarily
    measurements.sort_by_profile_and_point_id()

    # TODO: assumption that every profile has at least some measurements
    profiles = ProfileMeasurementGroup.split_by_profile_id(measurements)

    return profiles


def load_ZF_meta_from_ascii(meta_file_path: Union[str, Path]) -> Tuple[datetime, datetime]:
    meta_file_path = Path(meta_file_path)
    meta_data = meta_file_path.read_text()
    start_time_posix = float(re.findall("\d* = (\d+.\d+)\s*ms\s*\'Timestamp trigger\'", meta_data)[0]) / 1000
    stop_time_posix = float(re.findall("\d* = (\d+.\d+)\s*ms\s*\'Timestamp stop\'", meta_data)[0]) / 1000

    return datetime.utcfromtimestamp(start_time_posix), datetime.utcfromtimestamp(stop_time_posix)

