from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from dict_hash import sha256
import json
from pathlib import Path
import pickle
import re
from typing import Union, Tuple
from VM2dPLS.data import MeasurementGroupZF5016, ScanMeasurementGroup
from VM2dPLS.utility import timeit


class ScanHandler(ABC):

    def __init__(self, handler_args: dict):
        self._handler_args = handler_args
        self.measurement_file_path = Path(handler_args['filepath'])
        self.cache_interim = handler_args['cache_interim']
        self.restart_from_cache = handler_args['restart_from_cache']
        self.profiles = None

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def save_last_interim(self):
        pass

    @abstractmethod
    def load_from_interim(self):
        pass

    @abstractmethod
    def hash_run_parameters(self):
        pass


class ScanHandlerZF5016(ScanHandler):

    def __init__(self, handler_args: dict):
        super().__init__(handler_args)

        self.meta_file_path = Path(handler_args['filepath_meta'])
        if handler_args['time_window'] is not None:
            self.time_region_of_interest = tuple(timedelta(seconds=t) for t in handler_args['time_window'])
        else:
            self.time_region_of_interest = None

    def load_data(self, is_ascii: bool = True):
        ##TODO: Add possibility to start at specific intermediate step
        if self.restart_from_cache:
            last_step = self.load_from_last_interim()
        if not self.restart_from_cache or last_step < 0:
            last_step = 0
            if is_ascii:
                self.load_data_from_ascii()
            else:
                pass
        return last_step

    @timeit
    def load_data_from_ascii(self):
        assert self.meta_file_path is not None

        start_time, end_time = self.get_timestamps_from_meta()

        measurements = MeasurementGroupZF5016.from_ZF_ascii_file(self.measurement_file_path)

        measurements.profile_and_point_id_calculation_ZF()
        measurements.timestamp_interpolation(start_time, end_time)

        if self.time_region_of_interest is not None:
            if isinstance(self.time_region_of_interest[0], timedelta):
                self.time_region_of_interest = tuple(start_time + t for t in self.time_region_of_interest)
            measurements.time_filter(*self.time_region_of_interest)

        # TODO: Check if necessary..might cause runtime to explode unnecessarily
        measurements.sort_by_profile_and_point_id()

        # TODO: assumption that every profile has at least some measurements
        profiles = ScanMeasurementGroup.split_by_profile_id(measurements)

        # Remove start and end profile as they most likely are incomplete
        profile_keys = profiles.data.keys()
        first_profile_key, last_profile_key = min(profile_keys), max(profile_keys)

        profiles.remove_data_entries((first_profile_key, last_profile_key))

        # profiles.remove_profiles_with_fewer_points()

        self.profiles = profiles

        if self.cache_interim:
            self.save_last_interim(step_number=0)

    def get_timestamps_from_meta(self) -> Tuple[datetime, datetime]:
        meta_file_path = Path(self.meta_file_path)
        meta_data = meta_file_path.read_text()
        start_time_posix = float(re.findall("\d* = (\d+.\d+)\s*ms\s*\'Timestamp trigger\'", meta_data)[0]) / 1000
        stop_time_posix = float(re.findall("\d* = (\d+.\d+)\s*ms\s*\'Timestamp stop\'", meta_data)[0]) / 1000

        return datetime.utcfromtimestamp(start_time_posix), datetime.utcfromtimestamp(stop_time_posix)

    ## TODO: Define processing steps
    @timeit
    def save_last_interim(self, step_number: int = 0):
        if not self.cache_interim:
            return
        interim_results_folder = self.measurement_file_path.parent / self.measurement_file_path.stem / 'interim_results'
        interim_results_folder.mkdir(parents=True, exist_ok=True)
        config_hash = self.hash_run_parameters()

        if not list(interim_results_folder.glob('*' + str(config_hash))):
            config_files = list(interim_results_folder.glob('config_run_*'))
            already_used_numbers = [int(re.findall(r"config_run_(\d{3})", fn)[0])
                                    for fn in [str(f.stem) for f in config_files]]
            if not already_used_numbers:
                config_number = 0
            else:
                config_number = max(already_used_numbers) + 1
            config_folder = interim_results_folder / f"config_run_{config_number:03d}-{config_hash}"

            config_folder.mkdir()

            (config_folder / "run_parameters.txt").write_text(json.dumps(self._handler_args, indent=4))
        else:
            config_folder = list(interim_results_folder.glob('*' + str(config_hash)))[0]

        with (config_folder / f"{step_number:02d}.pkl").open('wb') as pkl_file:
            pickle.dump(self.profiles, pkl_file)

    def get_interim_config_folder(self) -> Path:
        interim_results_folder = self.measurement_file_path.parent / self.measurement_file_path.stem / 'interim_results'
        interim_results_folder.mkdir(parents=True, exist_ok=True)
        config_hash = self.hash_run_parameters()
        if list(interim_results_folder.glob('*' + str(config_hash))):
            config_folder = list(interim_results_folder.glob('*' + str(config_hash)))[0]
            return config_folder

    def load_from_last_interim(self) -> int:
        ## TODO: Catch if no intermediate available
        config_folder = self.get_interim_config_folder()
        if config_folder is not None:
            all_interim_steps = [int(p.stem) for p in config_folder.glob('[0-9]*.pkl')]
            last_step = max(all_interim_steps)

            self.load_from_interim(last_step)
            return last_step
        else:
            return -1



    @timeit
    def load_from_interim(self, step_number=0):
        interim_results_folder = self.measurement_file_path.parent / self.measurement_file_path.stem / 'interim_results'
        interim_results_folder.mkdir(parents=True, exist_ok=True)
        config_hash = self.hash_run_parameters()
        if list(interim_results_folder.glob('*' + str(config_hash))):
            config_folder = list(interim_results_folder.glob('*' + str(config_hash)))[0]
            with (config_folder / f"{step_number:02d}.pkl").open('rb') as pkl_file:
                self.profiles = pickle.load(pkl_file)

    def hash_run_parameters(self):
        relevant_parameters = ('filepath', 'scanner_type', 'time_window')
        return sha256({keys: self._handler_args[keys] for keys in relevant_parameters if keys in self._handler_args})
