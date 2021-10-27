from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd


@dataclass(init=True)
class MeasurementGroup:
    data: pd.DataFrame
    number_of_profiles: int
    number_of_points: int

    def timestamp_interpolation(self, start_time: datetime, end_time: datetime) -> None:
        # Assign time stamps per point based on `profile_id` and `v_angle`
        time_difference = end_time - start_time
        time_per_profile = time_difference / self.number_of_profiles
        time_per_deg = time_per_profile / 360

        self.data['timestamp'] = start_time + self.data['profile_id'] * time_per_profile + \
                                           self.data['v_angle'] * time_per_deg

    def time_filter(self, start_time: datetime, end_time: datetime, keep_original_copy: bool = False):
        # Filter by `time_region_of_interest`
        mask = (self.data['timestamp'] >= start_time) & (self.data['timestamp'] <= end_time)
        if keep_original_copy:
            self.data_original = self.copy

        self.data = self.data.loc[mask]

    def sort_by_profile_and_point_id(self):
        self.data.sort_values(['profile_id', 'point_id'], inplace=True)


class MeasurementGroupZF5016(MeasurementGroup):

    @classmethod
    def from_ZF_ascii_file(cls, measurement_file_path) -> 'MeasurementGroupZF5016':
        measurement_data = pd.read_csv(measurement_file_path, sep=';',
                                       names=['x', 'y', 'z', 'intensity', 'hz_angle', 'v_angle', 'range', 'col_id',
                                              'row_id'],
                                       dtype={'x': float, 'y': float, 'z': float, 'intensity': int, 'hz_angle': float,
                                              'v_angle': float, 'range': float, 'col_id': int, 'row_id': int})
        number_of_profiles = int((measurement_data['col_id'].max() + 1) / 2)
        number_of_points = int((measurement_data['row_id'].max() + 1) * 2)

        return cls(measurement_data, number_of_profiles, number_of_points)

    def profile_and_point_id_calculation_ZF(self) -> None:
        profile_id = np.mod(self.data['col_id'], self.number_of_profiles)
        right_hemisphere_flag = self.data['col_id'] // self.number_of_profiles
        point_id = (-1) ** (1 - right_hemisphere_flag) * self.data['row_id'] - (1 - right_hemisphere_flag) + \
                   self.number_of_points // 2
        self.data['profile_id'] = profile_id
        self.data['point_id'] = point_id


@dataclass
class ProfileMeasurementGroup:

    data: dict[int, MeasurementGroup]

    @classmethod
    def split_by_profile_id(cls, measurements: MeasurementGroup) -> 'ProfileMeasurementGroup':
        remaining_profile_ids = measurements.data['profile_id'].unique()
        profile_data = {}

        for _, profile_id in np.ndenumerate(remaining_profile_ids):
            profile_measurements = measurements.data[measurements.data['profile_id'] == profile_id]
            profile_data[profile_id] = MeasurementGroup(profile_measurements, 1, profile_measurements.shape[0])
        return cls(profile_data)
