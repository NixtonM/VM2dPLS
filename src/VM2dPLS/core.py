import logging
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Cursor
import numpy as np
from scipy import fftpack, signal
import sympy

import numpy.typing as npt
from typing import Optional
from VM2dPLS.data import MeasurementGroup, ScanMeasurementGroup
from VM2dPLS.io import ScanHandler
from VM2dPLS.least_squares import gauss_markov
from VM2dPLS.utility import convert_2d_polar_to_xy, convert_xy_to_2d_polar, timeit, convert_multi_2d_polar_to_xy

matplotlib.use('TkAgg')
theta_borders = []


class ProfileOutlierRemoval:
    def __init__(self, profile_handler: ScanHandler):
        self.profile_handler = profile_handler

    def remove_by_geometry(self, std: float = 2.0):
        pass


def select_vertical_angle_range_GUI(scan_handler: ScanHandler) -> Optional[np.ndarray]:
    ## TODO: add dependence on used scanner!
    display_profile_key = list(scan_handler.profiles.data.keys())[0]
    display_profile = scan_handler.profiles.data[display_profile_key].data

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location('S')
    ax.plot(display_profile['v_angle'], display_profile['range'],
            linestyle='None', marker='.', markersize='1', markerfacecolor='red')

    cursor = Cursor(ax, alpha=0.50, color='grey', linestyle='--')

    theta_borders = []

    def onclick(event):
        nonlocal theta_borders
        ##TODO: current assumption that the selected v_angle does NOT cross 360
        ##TODO: add line after first click
        theta_borders.append(np.mod(event.xdata, 360))

        if len(theta_borders) > 1:
            theta_borders = np.array(theta_borders, dtype=np.float64)
            theta_borders.sort()
            plt.close(fig)
        # print(event)

    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show(block=True)

    if not isinstance(theta_borders, list):
        return theta_borders


def cut_to_vertical_angle_range(scan_handler: ScanHandler, v_angle_start: float, v_angle_stop: float):
    assert v_angle_start <= v_angle_stop

    for measurement_group in scan_handler.profiles.data.values():
        measurement_group.v_angle_filter(v_angle_start, v_angle_stop)


def plot_profile(scan_handler: ScanHandler, profile_id: int = None):
    fig, ax = plt.subplots()

    if not profile_id:
        profile_id = list(scan_handler.profiles.data.keys())[0]
    measurement_group = scan_handler.profiles.data[profile_id]

    ax.plot(*convert_2d_polar_to_xy(*measurement_group.data[['range', 'v_angle']].to_numpy().transpose()),
            linestyle='None', marker='*', markersize='2')
    plt.show()


def plot_point_over_time(scan_handler: ScanHandler, point_id: int = None):
    if not point_id:
        point_id = 0

    profile_point_xyt = np.array([np.hstack((
        convert_multi_2d_polar_to_xy(*((np.array(profile_data.data[['range', 'v_angle']])).transpose())),
        profile_data.data[['timestamp']].to_numpy(dtype=np.float64) * 1e-9
    ))
        for profile_data in scan_handler.profiles.data.values()])

    profile_point_xyt[:, :, 2] = profile_point_xyt[:, :, 2] - profile_point_xyt[0, 0, 2]

    fig, ax = plt.subplots()
    ax.plot(*profile_point_xyt[:, point_id, (2, 1,)].transpose())



def animate_profiles(scan_handler: ScanHandler):
    fig, ax = plt.subplots()

    ln, = plt.plot([], [], linestyle='None', marker='.', markersize=2,
                   color='tab:red')

    def init():
        ax.set_xlim(-2, 35)
        ax.set_ylim(8.7, 8.82)
        return ln,

    def update(frame):
        x, y = convert_2d_polar_to_xy(
            *scan_handler.profiles.data[frame].data[['range', 'v_angle']].to_numpy().transpose()
        )
        ln.set_data(x, y)
        return ln,

    ani = FuncAnimation(fig, update, frames=list(scan_handler.profiles.data.keys()),
                        init_func=init, blit=True, interval=20, repeat=False)
    plt.show(block=True)


@timeit
def regional_binning(scan_handler: ScanHandler, target_bin_size: int = 100) -> ScanHandler:
    profile_master_key = list(scan_handler.profiles.data.keys())[0]
    profile_master = scan_handler.profiles.data[profile_master_key]
    number_of_bins = len(profile_master.data) // target_bin_size
    profile_split = np.array_split(np.array(profile_master.data['v_angle']), number_of_bins)
    bin_edges = [(v_angles[0], v_angles[-1]) for v_angles in profile_split]
    # profile_master_bins = [profile_master.get_mean_point(mask=profile_master.v_angle_mask(*be)) for be in bin_edges]
    profile_bins = {profile_id: [profile_data.get_mean_point(mask=profile_data.v_angle_mask(*be)) for be in bin_edges]
                    for (profile_id, profile_data) in scan_handler.profiles.data.items()}
    scan_handler.profiles = ScanMeasurementGroup.create_from_pandas_series(profile_bins)

    ##TODO: might need to cast to ScanHandlerZF5016; check if save_last_interim is generic (should be)
    scan_handler.save_last_interim(step_number=1)
