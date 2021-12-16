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
        profile_id = list(scan_handler.profiles.data.keys())[1]
    measurement_group = scan_handler.profiles.data[profile_id]

    ax.plot(*convert_2d_polar_to_xy(*measurement_group.data[['range', 'v_angle']].to_numpy().transpose()),
            linestyle='None', marker='*', markersize='2')
    plt.show()


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


def estimate_spatial_temporal_parameters(scan_handler: ScanHandler):

    # Extract binned points as 3d array (dim: profile, point, xyt)
    profile_point_xyt = np.array([np.hstack((
            convert_multi_2d_polar_to_xy(*((np.array(profile_data.data[['range', 'v_angle']])).transpose())),
            profile_data.data[['timestamp']].to_numpy(dtype=np.float64)*1e-9
        ))
        for profile_data in scan_handler.profiles.data.values()])

    # remove general bridge form
    mean_structure_form = np.median(profile_point_xyt, axis=0)
    profile_point_xyt_mean_removed = profile_point_xyt.copy()
    profile_point_xyt_mean_removed[:, :, 1] = profile_point_xyt[:, :, 1] - mean_structure_form[:, 1]

    # Step 2: Estimate 'temporal wave parameters' per point

    temporal_parameters = np.array(
        [np.array(estimate_temporal_parameters_v2(profile_xyt_per_point), dtype=np.float64)
         for profile_xyt_per_point in np.moveaxis(profile_point_xyt_mean_removed, 1, 0)]
    )

    _, zeta, freq_t, phi_t = np.nanmedian(temporal_parameters, axis=0)

    # Step 3: Estimate 'spatial wave parameters' per profile
    spatial_parameters = np.array([np.array(spatial_wave_Gauss_Markov(*profile_xy.transpose(),
                                                                      np.max(np.abs((profile_xy.transpose())[1])),
                                                                      1 / (np.max((profile_xy.transpose())[0]) - \
                                                                           np.min((profile_xy.transpose())[0]))),
                                            dtype=np.float64)
                                   for profile_xy in profile_point_xyt_mean_removed[:, :, 0:2]])

    # Select spatial median parameters on higher amplitudes
    ampli_cutoff = np.nanpercentile(spatial_parameters[:, 0], 90, interpolation='higher')
    top_spatial_parameters = spatial_parameters[spatial_parameters[:, 0] >= ampli_cutoff, :]

    ampli, freq_x, phi_x = np.nanmedian(top_spatial_parameters, axis=0)


    res = combined_wave_Gauss_Markov2(profile_point_xyt_mean_removed, ampli, zeta, freq_x, phi_x, freq_t, phi_t)

    print(1)


def estimate_temporal_parameters_v2(profile_xyt):
    ampli_0, freq_0 = calculate_starting_values_with_fft(*profile_xyt[:, (2, 1)].transpose())
    return temporal_wave_Gauss_Markov(*profile_xyt[:, 1:].transpose(), ampli_0, freq_0)


def calculate_starting_values_with_fft(t, data_y):
    # TODO: Settings for highpass filter
    f_HPF = 1
    N_HPF = 2

    if (data_y.shape[0] % 2) == 1:
        data_y = data_y[0:-1]
        t = t[0:-1]

    N = data_y.shape[0]
    T = (t[-1] - t[0]) / N
    fs = 1 / T

    y = data_y - data_y.mean()

    # yf = fftpack.fft(y)
    xf = fftpack.fftfreq(N, T)


    b_high, a_high = signal.butter(N_HPF, f_HPF / (fs / 2), btype="highpass", analog=False, output='ba')
    y_filtered_high = signal.lfilter(b_high, a_high, y)
    yf_filtered_high = fftpack.fft(y_filtered_high)

    freq = xf[np.argmax(np.abs(yf_filtered_high[:N // 2]))]
    ampli = np.max(np.abs(yf_filtered_high[:N // 2]))

    return ampli, freq


def estimate_temporal_parameters(scan_handler: ScanHandler, profile_point: int, plot_fft=False):
    point_xy = np.array([convert_2d_polar_to_xy(
        *scan_handler.profiles.data[k].data.loc[profile_point, ['range', 'v_angle']])
        for k in scan_handler.profiles.data.keys()]
    ).transpose()

    data_y = point_xy[1]
    timings = [scan_handler.profiles.data[k].data.loc[profile_point, 'timestamp']
               for k in scan_handler.profiles.data.keys()]

    if (data_y.shape[0] % 2) == 1:
        data_y = data_y[0:-1]
        timings = timings[0:-1]

    N = data_y.shape[0]
    T = ((timings[-1] - timings[0]).to_numpy().__float__() * 1e-9) / N
    fs = 1 / T

    y = data_y - data_y.mean()
    t = np.array([t.to_numpy().__float__() * 1e-9 for t in timings])
    # t_reduced = t - t[0]
    #
    # y = y[t_reduced > 3]
    # t_reduced = t_reduced[t_reduced > 3]

    yf = fftpack.fft(y)
    xf = fftpack.fftfreq(N, T)

    f_HPF = 1
    N_HPF = 2

    b_high, a_high = signal.butter(N_HPF, f_HPF / (fs / 2), btype='highpass', analog=False, output='ba')
    y_filtered_high = signal.lfilter(b_high, a_high, y)
    yf_filtered_high = fftpack.fft(y_filtered_high)


    # xf = fftpack.fftshift(xf[1:])

    # yplot = fftpack.fftshift(yf[1:])

    # y_normed = 2 * (y - y.min()) / (y.max() - y.min()) - 1
    # yf_normed = fftpack.fft(y_normed)
    if plot_fft:
        fig, axs = plt.subplots(4)

        axs[0].plot(t, y)
        axs[0].set_ylabel('original signal')
        axs[1].plot(xf[:N // 2], 2.0 / N * np.abs(yf[:N // 2]))
        axs[1].set_ylabel('FFT on original signal')
        axs[2].plot(t, y_filtered_high)
        axs[2].set_ylabel('Highpass filtered signal')
        axs[3].plot(xf[:N // 2], 2.0 / N * np.abs(yf_filtered_high[:N // 2]))
        axs[3].set_ylabel('Highpass filtered signal')


        # axs[4].plot(t, y_filtered_band)
        # axs[5].plot(xf[:N // 2], 2.0 / N * np.abs(yf_filtered_band[:N // 2]))

        plt.show(block=False)

    freq_1 = xf[np.argmax(np.abs(yf_filtered_high[:N // 2]))]
    A_1 = np.max(np.abs(yf_filtered_high[:N // 2]))

    # bandpass_filter = signal.butter(N_HPF, [(freq_1 - 0.5) / (fs / 2), (freq_1 - 0.5) / (fs / 2)], btype='bandpass', analog=False, output='ba')
    # y_filtered_band = signal.lfilter(b_band, a_band, y)
    # yf_filtered_band = fftpack.fft(y_filtered_band)

    results_GM = temporal_wave_Gauss_Markov(y_filtered_high, t, A_1, freq_1)

    return results_GM


def estimate_spatial_parameters(scan_handler: ScanHandler):
    temporal_spatial_xy = np.array([convert_multi_2d_polar_to_xy(
        *((np.array(scan_handler.profiles.data[k].data[['range', 'v_angle']])).transpose()))
        for k in scan_handler.profiles.data.keys()])

    # TODO: Check if valid approach
    mean_structure_form = np.median(temporal_spatial_xy, axis=0)
    temporal_spatial_xy_mean_removed = temporal_spatial_xy.copy()
    temporal_spatial_xy_mean_removed[:, :, 1] = temporal_spatial_xy_mean_removed[:, :, 1] - mean_structure_form[:, 1]

    spatial_parameters = np.array([np.array(spatial_wave_Gauss_Markov(*profile_xy.transpose(),
                                                                      np.max(np.abs((profile_xy.transpose())[1])),
                                                                      1/(np.max((profile_xy.transpose())[0]) -\
                                                                         np.min((profile_xy.transpose())[0]))),
                                   dtype=np.float64)
                                   for profile_xy in temporal_spatial_xy_mean_removed])
    print(1)


def spatial_wave_Gauss_Markov(x_pos, y, ampli_0=0.1, freq_0=0.1, phi_0=0):

    ampli, freq, phi, x = sympy.symbols('A freq phi x')
    s = (ampli, freq, phi, x)

    sympy.init_printing(use_unicode=True)

    f = ampli*sympy.sin(2 * sympy.pi * freq * x + phi)
    f_dampli = sympy.diff(f, ampli)
    f_dfreq = sympy.diff(f, freq)
    f_dphi = sympy.diff(f, phi)

    partial_derivatives = sympy.Matrix([[f_dampli, f_dfreq, f_dphi]])

    observation_func = sympy.lambdify(s, f, modules='numpy')

    P = np.eye(len(y))

    partial_derivatives_func = sympy.lambdify(s, partial_derivatives, modules='numpy')

    for j in range(100):
        L_0 = observation_func(ampli_0, freq_0, phi_0, x_pos)
        l_reduced = y - L_0

        A = np.zeros((0, 3,), dtype=float)
        for i in range(len(y)):
            A = np.vstack((A, partial_derivatives_func(ampli_0, freq_0, phi_0, x_pos[i])))

        N = A.transpose() @ P @ A

        try:
            dx = np.linalg.inv(N) @ A.transpose() @ l_reduced
            ampli_i, freq_i, phi_i = (np.array([ampli_0, freq_0, phi_0]) + dx)

            # Numeric optimization
            if freq_i < 0:
                freq_i = np.abs(freq_i)
                ampli_i = - ampli_i
            if ampli_i < 0:
                ampli_i = np.abs(ampli_i)
                phi_i = phi_i + np.pi
            phi_i = np.mod(phi_i, 2 * np.pi)


            v = A @ dx - l_reduced
            L_i = L_0 + A @ np.linalg.inv(N) @ A.transpose() @ P @ l_reduced
            ampli_0, freq_0, phi_0 = ampli_i, freq_i, phi_i
            # print(f"Step {j:3d}:\t\tAmpl: {ampli_i:+.6e}\tFreq: {freq_i:+.6e}\tPhi: {phi_i:+.6e}")

            # TODO: Better stop criterion
            if np.linalg.norm(dx) < 0.001:
                # print(f"Step {j:3d}:\t\tAmpl: {ampli_i:+.6e}\tFreq: {freq_i:+.6e}\tPhi: {phi_i:+.6e}")
                return ampli_0, freq_0, phi_0

        except np.linalg.LinAlgError as err:
            if 'Singular matrix' in str(err):
                logger = logging.getLogger(__name__)
                logger.warning(f"Iteration {j} produces singular N matrix.")
                break
            else:
                raise
    if j >= 999:
        logger = logging.getLogger(__name__)
        logger.warning(f"Doesn't converge.")
    return (np.nan,)*3







# def phase_IQ(signal_data, step_times, freq):
#     freq_LPF = 1
#
#     freq_sampling = 1 / ((step_times[-1] - step_times[0]) / (len(step_times)-1))
#
#     I = np.cos(2 * np.pi * freq * step_times)
#     Q = np.sin(2 * np.pi * freq * step_times)
#
#     I_signal = signal_data * I
#     Q_signal = signal_data * Q
#
#     low_pass_filter = signal.butter(2, freq_LPF / (freq_sampling / 2), btype='low', analog=False, output='ba')
#
#     I_signal_LPF = signal.lfilter(*low_pass_filter, I_signal)
#     Q_signal_LPF = signal.lfilter(*low_pass_filter, Q_signal)
#
#     phase = np.arctan2(I_signal_LPF, Q_signal_LPF)
#     # mod_signal = np.mean(np.sqrt(np.square(I_signal_LPF) + np.square(Q_signal_LPF)))
#
#     return phase


def temporal_wave_Gauss_Markov(L, timing, ampli_0, freq_0, zeta_0=1e-2, phi_0=0):
    ampli, zeta, freq, phi, t = sympy.symbols('A zeta freq phi t')
    s = (ampli, zeta, freq, phi, t)

    sympy.init_printing(use_unicode=True)

    f = ampli * sympy.exp(-zeta * t) * sympy.sin(2 * sympy.pi * freq * t + phi)
    f_dampli = sympy.diff(f, ampli)
    f_dzeta = sympy.diff(f, zeta)
    f_dfreq = sympy.diff(f, freq)
    f_dphi = sympy.diff(f, phi)

    partial_derivatives = sympy.Matrix([[f_dampli, f_dzeta, f_dfreq, f_dphi]])

    observation_func = sympy.lambdify(s, f, modules='numpy')


    P = np.eye(len(L))

    partial_derivatives_func = sympy.lambdify(s, partial_derivatives, modules='numpy')

    for j in range(1000):
        L_0 = observation_func(ampli_0, zeta_0, freq_0, phi_0, timing)
        l_reduced = L - L_0

        A = np.zeros((0, 4,), dtype=float)
        for i in range(len(L)):
            A = np.vstack((A, partial_derivatives_func(ampli_0, zeta_0, freq_0, phi_0, timing[i])))

        N = A.transpose() @ P @ A

        try:
            dx = np.linalg.inv(N) @ A.transpose() @ l_reduced
            ampli_i, zeta_i, freq_i, phi_i = (np.array([ampli_0, zeta_0, freq_0, phi_0]) + dx)

            # Numeric optimization
            if freq_i < 0:
                freq_i = np.abs(freq_i)
                ampli_i = - ampli_i
            if ampli_i < 0:
                ampli_i = np.abs(ampli_i)
                phi_i = phi_i + np.pi
            phi_i = np.mod(phi_i, 2 * np.pi)

            if zeta_i < 0:
                zeta_i = 0


            v = A @ dx - l_reduced
            L_i = L_0 + A @ np.linalg.inv(N) @ A.transpose() @ P @ l_reduced
            ampli_0, zeta_0, freq_0, phi_0 = ampli_i, zeta_i, freq_i, phi_i

            # TODO: Better stop criterion
            if np.linalg.norm(dx) < 0.001:
                print(f"Step {j:3d}:\tAmpl: {ampli_i:+.6e}\tZeta: {zeta_i:+.6e}\tFreq: {freq_i:+.6e}\tPhi: {phi_i:+.6e}")
                return ampli_0, zeta_0, freq_0, phi_0

        except np.linalg.LinAlgError as err:
            if 'Singular matrix' in str(err):
                logger = logging.getLogger(__name__)
                logger.warning(f"Iteration {j} produces singular N matrix.")
                break
            else:
                raise
    if j == 999:
        logger = logging.getLogger(__name__)
        logger.warning(f"Doesn't converge.")
    return (np.nan,)*4


def combined_wave_Gauss_Markov(profile_point_xyt, ampli_0, zeta_0, freq_x_0, phi_x_0, freq_t_0, phi_t_0):

    x_values = np.ravel(profile_point_xyt[:, :, 0])
    t_values = np.ravel(profile_point_xyt[:, :, 2])

    L = np.ravel(profile_point_xyt[:, :, 1])

    ampli, zeta, freq_x, phi_x, x, freq_t, phi_t, t = sympy.symbols('A zeta freq_x phi_x x freq_t phi_t t')
    s = (ampli, zeta, freq_x, phi_x, x, freq_t, phi_t, t)

    # sympy.init_printing(use_unicode=True)

    f = ampli * sympy.exp(-zeta * t) * sympy.sin(2 * sympy.pi * freq_x * x + phi_x) * \
        sympy.sin(2 * sympy.pi * freq_t * t + phi_t)

    f_dampli = sympy.diff(f, ampli)
    f_dzeta = sympy.diff(f, zeta)
    f_dfreq_t = sympy.diff(f, freq_t)
    f_dphi_t = sympy.diff(f, phi_t)
    f_dfreq_x = sympy.diff(f, freq_x)
    f_dphi_x = sympy.diff(f, phi_x)

    partial_derivatives = sympy.Matrix([[f_dampli, f_dzeta, f_dfreq_x, f_dphi_x, f_dfreq_t, f_dphi_t]])

    observation_func = sympy.lambdify(s, f, modules='numpy')

    nbObs = len(L)

    P = np.eye(nbObs)

    partial_derivatives_func = sympy.lambdify(s, partial_derivatives, modules='numpy')


    for j in range(1000):

        L_0 = observation_func(ampli_0, zeta_0, freq_x_0, phi_x_0, x_values, freq_t_0, phi_t_0, t_values)
        l_reduced = L - L_0

        # A = np.squeeze(partial_derivatives_func(ampli_0, zeta_0, freq_x_0, phi_x_0, x_values, freq_t_0,
        #                                         phi_t_0, t_values)).transpose()

        A = np.zeros((0, 6,), dtype=float)
        for i in range(len(L)):
            A = np.vstack((A, partial_derivatives_func(ampli_0, zeta_0, freq_x_0, phi_x_0, x_values[i], freq_t_0,
                                                       phi_t_0, t_values[i])))

        N = A.transpose() @ P @ A

        try:
            dx = np.linalg.inv(N) @ A.transpose() @ l_reduced
            ampli_i, zeta_i, freq_x_i, phi_x_i, freq_t_i, phi_t_i = (np.array([ampli_0, zeta_0, freq_x_0, phi_x_0,
                                                                               freq_t_0, phi_t_0]) + dx)
            # ampli_i, zeta_i, freq_i, phi_i = (np.array([ampli_0, zeta_0, freq_0, phi_0]) + dx)

            # Numeric optimization
            if freq_x_i < 0:
                freq_x_i = np.abs(freq_x_i)
                ampli_i = - ampli_i
            if freq_t_i < 0:
                freq_t_i = np.abs(freq_t_i)
                ampli_i = - ampli_i

            if ampli_i < 0:
                ampli_i = np.abs(ampli_i)
                phi_t_i = phi_t_i + np.pi
            phi_t_i = np.mod(phi_t_i, 2 * np.pi)
            phi_x_i = np.mod(phi_x_i, 2 * np.pi)

            if zeta_i < 0:
                zeta_i = 0


            v = A @ dx - l_reduced
            # L_i = L_0 + A @ np.linalg.inv(N) @ A.transpose() @ P @ l_reduced
            ampli_0, zeta_0, freq_x_0, phi_x_0, freq_t_0, phi_t_0 = ampli_i, zeta_i, freq_x_i, phi_x_i, freq_t_i, phi_t_i

            print(
                f"Step {j:3d}:\tAmpl: {ampli_i:+.4e}\tZeta: {zeta_i:+.4e}\tf_x: {freq_x_i:+.4e}\tphi_x: {phi_x_i:+.4e}\tf_t: {freq_x_i:+.4e}\tphi_t: {phi_t_i:+.4e}")

            # TODO: Better stop criterion
            if np.linalg.norm(dx) < 0.001:
                # print(f"Step {j:3d}:\tAmpl: {ampli_i:+.6e}\tZeta: {zeta_i:+.6e}\tf_x: {freq_x_i:+.6e}\tphi_x: {phi_x_i:+.6e}\tf_t: {freq_x_i:+.6e}\tphi_t: {phi_t_i:+.6e}")
                return ampli_0, zeta_0, freq_x_0, phi_x_0, freq_t_0, phi_t_0

        except np.linalg.LinAlgError as err:
            if 'Singular matrix' in str(err):
                logger = logging.getLogger(__name__)
                logger.warning(f"Iteration {j} produces singular N matrix.")
                break
            else:
                raise
    if j == 999:
        logger = logging.getLogger(__name__)
        logger.warning(f"Doesn't converge.")
    return (np.nan,)*6


def combined_wave_Gauss_Markov2(profile_point_xyt, ampli_0, zeta_0, freq_x_0, phi_x_0, freq_t_0, phi_t_0):

    x_values = np.ravel(profile_point_xyt[:, :, 0])
    t_values = np.ravel(profile_point_xyt[:, :, 2])

    L = np.ravel(profile_point_xyt[:, :, 1])

    ampli, freq_x, phi_x, x, freq_t, phi_t, t = sympy.symbols('A freq_x phi_x x freq_t phi_t t')
    s = (ampli, freq_x, phi_x, x, freq_t, phi_t, t)

    # sympy.init_printing(use_unicode=True)

    f = ampli * sympy.sin(2 * sympy.pi * freq_x * x + phi_x) * \
        sympy.sin(2 * sympy.pi * freq_t * t + phi_t)

    f_dampli = sympy.diff(f, ampli)
    f_dfreq_t = sympy.diff(f, freq_t)
    f_dphi_t = sympy.diff(f, phi_t)
    f_dfreq_x = sympy.diff(f, freq_x)
    f_dphi_x = sympy.diff(f, phi_x)

    partial_derivatives = sympy.Matrix([[f_dampli, f_dfreq_x, f_dphi_x, f_dfreq_t, f_dphi_t]])

    observation_func = sympy.lambdify(s, f, modules='numpy')

    nbObs = len(L)

    P = np.eye(nbObs)

    partial_derivatives_func = sympy.lambdify(s, partial_derivatives, modules='numpy')


    for j in range(1000):

        L_0 = observation_func(ampli_0, freq_x_0, phi_x_0, x_values, freq_t_0, phi_t_0, t_values)
        l_reduced = L - L_0

        A = np.squeeze(partial_derivatives_func(ampli_0, freq_x_0, phi_x_0, x_values, freq_t_0,
                                                phi_t_0, t_values)).transpose()

        # A = np.zeros((0, 5,), dtype=float)
        # for i in range(len(L)):
        #     A = np.vstack((A, partial_derivatives_func(ampli_0, freq_x_0, phi_x_0, x_values[i], freq_t_0,
        #                                                phi_t_0, t_values[i])))

        N = A.transpose() @ P @ A

        try:
            dx = np.linalg.inv(N) @ A.transpose() @ l_reduced
            ampli_i, freq_x_i, phi_x_i, freq_t_i, phi_t_i = (np.array([ampli_0, freq_x_0, phi_x_0,
                                                                               freq_t_0, phi_t_0]) + dx)
            # ampli_i, zeta_i, freq_i, phi_i = (np.array([ampli_0, zeta_0, freq_0, phi_0]) + dx)

            # Numeric optimization
            if freq_x_i < 0:
                freq_x_i = np.abs(freq_x_i)
                ampli_i = - ampli_i
                
            if freq_t_i < 0:
                freq_t_i = np.abs(freq_t_i)
                ampli_i = - ampli_i

            if ampli_i < 0:
                ampli_i = np.abs(ampli_i)
                phi_t_i = phi_t_i + np.pi
            phi_t_i = np.mod(phi_t_i, 2 * np.pi)
            phi_x_i = np.mod(phi_x_i, 2 * np.pi)



            v = A @ dx - l_reduced
            # L_i = L_0 + A @ np.linalg.inv(N) @ A.transpose() @ P @ l_reduced
            ampli_0, freq_x_0, phi_x_0, freq_t_0, phi_t_0 = ampli_i, freq_x_i, phi_x_i, freq_t_i, phi_t_i

            print(
                f"Step {j:3d}:\tAmpl: {ampli_i:+.4e}\tf_x: {freq_x_i:+.4e}\tphi_x: {phi_x_i:+.4e}\tf_t: {freq_x_i:+.4e}\tphi_t: {phi_t_i:+.4e}")

            # TODO: Better stop criterion
            if np.linalg.norm(dx) < 0.001:
                # print(f"Step {j:3d}:\tAmpl: {ampli_i:+.6e}\tZeta: {zeta_i:+.6e}\tf_x: {freq_x_i:+.6e}\tphi_x: {phi_x_i:+.6e}\tf_t: {freq_x_i:+.6e}\tphi_t: {phi_t_i:+.6e}")
                return ampli_0, freq_x_0, phi_x_0, freq_t_0, phi_t_0

        except np.linalg.LinAlgError as err:
            if 'Singular matrix' in str(err):
                logger = logging.getLogger(__name__)
                logger.warning(f"Iteration {j} produces singular N matrix.")
                break
            else:
                raise
    if j == 999:
        logger = logging.getLogger(__name__)
        logger.warning(f"Doesn't converge.")
    return (np.nan,)*5
