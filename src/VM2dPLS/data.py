from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd
from typing import Type, List, Dict
from VM2dPLS.utility import reject_outliers, convert_2d_polar_to_xy, convert_xy_to_2d_polar


@dataclass(init=True)
class MeasurementGroup:
    ##TODO: Reevaluate number_of_points!
    data: pd.DataFrame

    @classmethod
    def create_from_pandas_series(cls, series: list[pd.Series]):
        data = pd.concat(series, axis=1).transpose()
        return cls(data)

    def time_filter(self, start_time: datetime, end_time: datetime, keep_original_copy: bool = False):
        # Filter by `time_region_of_interest`
        mask = (self.data['timestamp'] >= start_time) & (self.data['timestamp'] <= end_time)
        if keep_original_copy:
            self.data_original = self.copy

        self.data = self.data.loc[mask]

    def reduce_to_start_time(self, start_time : datetime):
        self.data['timestamp'] = self.data['timestamp'] - start_time

    def sort_by_profile_and_point_id(self):
        self.data.sort_values(['profile_id', 'point_id'], inplace=True)

    def v_angle_filter(self, v_angle_start: float, v_angle_stop: float):
        # TODO: Assumption about turning direction and coordinate system made..Restate more genereally Recalculate
        self.data = self.data.loc[self.v_angle_mask(v_angle_start, v_angle_stop)]

    def v_angle_mask(self, v_angle_start: float, v_angle_stop: float):
        return (self.data['v_angle'] >= v_angle_start) & (self.data['v_angle'] <= v_angle_stop)

    def get_mean_point(self, mask=None):
        if mask is None:
            mask = np.full((self.data.shape[0],), True, dtype=bool)

        x, y = convert_2d_polar_to_xy(*self.data.loc[mask, ['range', 'v_angle']].to_numpy().transpose())
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        distance_mean, v_angle_mean = convert_xy_to_2d_polar(x_mean, y_mean)

        timestamp_mean = np.mean(self.data.loc[mask, 'timestamp'])

        intensity_mean = np.int32(np.mean(self.data.loc[mask, 'intensity']))

        series_mean = pd.Series(data={'range': distance_mean, 'v_angle': v_angle_mean, 'timestamp': timestamp_mean,
                                      'intensity': intensity_mean})
        return series_mean



@dataclass(init=False)
class MeasurementGroupZF5016(MeasurementGroup):

    number_of_profiles: int
    unique_v_angles_count: int

    def __init__(self, measurement_data, number_of_profiles, unique_v_angles_count):
        super().__init__(measurement_data)
        self.number_of_profiles = number_of_profiles
        self.unique_v_angles_count = unique_v_angles_count

    @classmethod
    def from_ZF_ascii_file(cls, measurement_file_path) -> 'MeasurementGroupZF5016':
        measurement_data = pd.read_csv(measurement_file_path, sep=';', header=None, index_col=False,
                                       names=['x', 'y', 'z', 'intensity', 'hz_angle', 'v_angle', 'range', 'col_id',
                                              'row_id'],
                                       usecols=['v_angle', 'range', 'intensity', 'col_id', 'row_id'],
                                       dtype={'v_angle': np.float64, 'range': np.float64, 'intensity': np.int32,
                                              'col_id': np.int32, 'row_id': np.int32})
        measurement_data.v_angle = np.deg2rad(measurement_data.v_angle)
        number_of_profiles = int((measurement_data['col_id'].max() + 1) / 2)
        unique_v_angles_count = int((measurement_data['row_id'].max() + 1) * 2)

        return cls(measurement_data, number_of_profiles, unique_v_angles_count)

    def profile_and_point_id_calculation_ZF(self) -> None:
        profile_id = np.mod(self.data['col_id'], self.number_of_profiles)
        right_hemisphere_flag = self.data['col_id'] // self.number_of_profiles
        point_id = (-1) ** (1 - right_hemisphere_flag) * self.data['row_id'] - (1 - right_hemisphere_flag) + \
                   self.unique_v_angles_count // 2
        self.data['profile_id'] = profile_id
        self.data['point_id'] = point_id
        self.data.drop(columns=['col_id', 'row_id'])

    def timestamp_interpolation(self, start_time: datetime, end_time: datetime) -> None:
        # Assign time stamps per point based on `profile_id` and `v_angle`
        time_difference = end_time - start_time
        time_per_profile = time_difference / self.number_of_profiles
        time_per_deg = time_per_profile / (2*np.pi)

        self.data['timestamp'] = start_time + self.data['profile_id'] * time_per_profile + \
                                           self.data['v_angle'] * time_per_deg

@dataclass
class ScanMeasurementGroup:

    data: dict[int, MeasurementGroup]

    @classmethod
    def split_by_profile_id(cls, measurements: Type[MeasurementGroup]) -> 'ScanMeasurementGroup':
        remaining_profile_ids = measurements.data['profile_id'].unique()
        profile_data = {}

        for _, profile_id in np.ndenumerate(remaining_profile_ids):
            profile_measurements = measurements.data[measurements.data['profile_id'] == profile_id]
            profile_data[profile_id] = MeasurementGroup(profile_measurements)
        return cls(profile_data)

    @classmethod
    def create_from_pandas_series(cls, series: dict[int, list[pd.Series]]):
        profile_data = {profile_id: MeasurementGroup.create_from_pandas_series(binned_profile)
                        for (profile_id, binned_profile) in series.items()}
        return cls(profile_data)

    def remove_data_entries(self, keys):
        for key in keys:
            if key in self.data.keys():
                del self.data[key]

    # def remove_profiles_with_fewer_points(self):
    #     point_counts = np.array([(p_k, len(p_mg.data)) for p_k, p_mg in self.data.items()]).swapaxes(0, 1)
    #     mask = reject_outliers(point_counts[1])
    #     for i, m in enumerate(mask):
    #         if not m:
    #             del self.data[point_counts[0][i]]



