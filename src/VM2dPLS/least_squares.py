import logging

import numpy as np
from scipy import fftpack, signal
import sympy

from VM2dPLS.io import ScanHandler
from VM2dPLS.utility import convert_multi_2d_polar_to_xy


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
        [np.array(estimate_temporal_parameters(profile_xyt_per_point), dtype=np.float64)
         for profile_xyt_per_point in np.moveaxis(profile_point_xyt_mean_removed, 1, 0)]
    )

    # Remove NaN entries
    temporal_parameters = temporal_parameters[~np.isnan(temporal_parameters[:, 0])]

    temporal_parameters[:, (0, 2, 3)] = numeric_wave(temporal_parameters[:, (0, 2, 3)])

    _, zeta, freq_t, phi_t = np.nanmedian(temporal_parameters, axis=0)

    # Step 3: Estimate 'spatial wave parameters' per profile

    mean_time = np.median(profile_point_xyt_mean_removed[:, :, 2], axis=1)
    # sin_factor = np.sin(2 * np.pi * freq_t * mean_time + phi_t)
    # sin_mask = sin_factor > 0.8
    time_factor = np.exp(-zeta * mean_time) * np.sin(2 * np.pi * freq_t * mean_time + phi_t)
    time_mask = time_factor >= np.percentile(time_factor, 90)

    spatial_parameters = np.array([np.array(spatial_wave_Gauss_Markov(profile_xy,
                                                                      np.max(np.abs((profile_xy.transpose())[1])),
                                                                      1 / (2*(np.max((profile_xy.transpose())[0]) -
                                                                           np.min((profile_xy.transpose())[0])))),
                                            dtype=np.float64)
                                   for profile_xy in profile_point_xyt_mean_removed[time_mask, :, :]])

    # Remove NaN entries
    spatial_parameters = spatial_parameters[~np.isnan(spatial_parameters[:, 0])]

    spatial_parameters = numeric_wave(spatial_parameters)

    ampli, freq_x, phi_x = np.median(spatial_parameters, axis=0)

    results = combined_wave_Gauss_Markov(profile_point_xyt_mean_removed, ampli, zeta, freq_x, phi_x, freq_t, phi_t)

    profile_point_xrt = calculate_residuals(profile_point_xyt_mean_removed, *results)

    # temporal_parameters2 = np.array(
    #     [np.array(estimate_temporal_parameters(profile_xrt_per_point), dtype=np.float64)
    #      for profile_xrt_per_point in np.moveaxis(profile_point_xrt, 1, 0)]
    # )
    #
    # # Remove NaN entries
    # temporal_parameters2 = temporal_parameters2[~np.isnan(temporal_parameters2[:, 0])]
    #
    # temporal_parameters2[:, (0, 2, 3)] = numeric_wave(temporal_parameters2[:, (0, 2, 3)])
    #
    # amplitude_mask = temporal_parameters2[:, 0] >= np.percentile(temporal_parameters2[:, 0], 90)
    #
    # _, zeta2, freq_t2, phi_t2 = np.nanmedian(temporal_parameters2[amplitude_mask, :], axis=0)
    #
    # median_time2 = np.median(profile_point_xrt[:, :, 2], axis=1)
    # # sin_factor = np.sin(2 * np.pi * freq_t * mean_time + phi_t)
    # # sin_mask = sin_factor > 0.8
    # time_factor2 = np.exp(-zeta2 * median_time2) * np.sin(2 * np.pi * freq_t2 * median_time2 + phi_t2)
    # time_mask2 = time_factor2 >= np.percentile(time_factor2, 90)
    #
    # spatial_parameters2 = np.array([np.array(spatial_wave_Gauss_Markov(profile_xy,
    #                                                                   np.max(np.abs((profile_xy.transpose())[1])),
    #                                                                   1 / ((np.max((profile_xy.transpose())[0]) -
    #                                                                             np.min((profile_xy.transpose())[0])))),
    #                                         dtype=np.float64)
    #                                for profile_xy in profile_point_xrt[time_mask2, :, :]])
    # spatial_parameters2 = spatial_parameters2[~np.isnan(spatial_parameters2[:, 0])]
    #
    # spatial_parameters2 = numeric_wave(spatial_parameters2)
    #
    # ampli2, freq_x2, phi_x2 = np.median(spatial_parameters2, axis=0)
    #
    # results2 = combined_wave_Gauss_Markov(profile_point_xrt, ampli2, zeta2, freq_x2, phi_x2, freq_t2, phi_t2)

    return results


def calculate_residuals(profile_point_data, ampli, zeta, freq_x, phi_x, freq_t, phi_t):
    x = np.ravel(profile_point_data[:, :, 0])
    t = np.ravel(profile_point_data[:, :, 2])

    residuals = (ampli * np.exp(-zeta * t) * np.sin(2 * np.pi * freq_x * x + phi_x) *
                 np.sin(2 * np.pi * freq_t * t + phi_t)) - np.ravel(profile_point_data[:, :, 1])

    profile_point_data[:, :, 1] = np.reshape(residuals, profile_point_data.shape[0:2])

    return profile_point_data


def numeric_wave(afp):
    # Flip negative frequency by flipping amplitude and phase shift
    afp[:, 0] = afp[:, 0] * np.sign(afp[:, 1])
    afp[:, 2] = afp[:, 2] * np.sign(afp[:, 1])

    afp[:, 1] = np.abs(afp[:, 1])

    # Flip negative amplitude by adding pi to phase shift
    afp[:, 2] = (afp[:, 0] < 0) * np.pi + afp[:, 2]

    afp[:, 0] = np.abs(afp[:, 0])

    # Remove extra phase cycles
    afp[:, 2] = np.mod(afp[:, 2], 2 * np.pi)

    return afp


def estimate_temporal_parameters(profile_xyt):
    ampli_0, freq_0 = calculate_starting_values_with_fft(*profile_xyt[:, (2, 1)].transpose())
    return temporal_wave_Gauss_Markov(profile_xyt, ampli_0, freq_0)


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

def combined_wave_Gauss_Markov(profile_point_xyt_mean_removed, ampli_0, zeta_0, freq_x_0, phi_x_0, freq_t_0, phi_t_0):
    x_sym = sympy.symbols('A zeta f_x phi_x f_t phi_t')
    x_0 = (ampli_0, zeta_0, freq_x_0, phi_x_0, freq_t_0, phi_t_0)
    y_sym = sympy.symbols('x t')
    f_sym = sympy.Matrix([x_sym[0] * sympy.exp(-x_sym[1] * y_sym[1]) *
                          sympy.sin(2*sympy.pi*x_sym[2]*y_sym[0] + x_sym[3]) *
                          sympy.sin(2*sympy.pi*x_sym[4]*y_sym[1] + x_sym[5])])

    L = np.ravel(profile_point_xyt_mean_removed[:, :, 1])

    outlier_mask = np.abs(L) < 0.1
    L = L[outlier_mask]

    y_val = (np.ravel(profile_point_xyt_mean_removed[:, :, 0])[outlier_mask],
             np.ravel(profile_point_xyt_mean_removed[:, :, 2])[outlier_mask])

    return gauss_markov(x_sym, y_sym, f_sym, L, x_0, y_val)


def temporal_wave_Gauss_Markov(profile_xyt, ampli_0, freq_0, zeta_0=1e-2, phi_0=0):
    x_sym = sympy.symbols('A zeta freq phi')
    x_0 = (ampli_0, zeta_0, freq_0, phi_0)
    y_sym = (sympy.symbols('t'), )

    f_sym = sympy.Matrix([x_sym[0] * sympy.exp(-x_sym[1] * y_sym[0]) *
                          sympy.sin(2 * sympy.pi * x_sym[2] * y_sym[0] + x_sym[3])])

    L = np.squeeze(profile_xyt[:, 1])
    y_val = (np.squeeze(profile_xyt[:, 2]), )

    return gauss_markov(x_sym, y_sym, f_sym, L, x_0, y_val)


def spatial_wave_Gauss_Markov(point_xyt, ampli_0=0.1, freq_0=0.1, phi_0=0):
    x_sym = sympy.symbols('A freq phi')
    x_0 = (ampli_0, freq_0, phi_0)
    y_sym = (sympy.symbols('x'),)

    f_sym = sympy.Matrix([x_sym[0] * sympy.sin(2 * sympy.pi * x_sym[1] * y_sym[0] + x_sym[2])])

    L = np.squeeze(point_xyt[:, 1])
    y_val = (np.squeeze(point_xyt[:, 0]), )

    return gauss_markov(x_sym, y_sym, f_sym, L, x_0, y_val)


def gauss_markov(x_symbols: tuple[sympy.Symbol, ...], y_symbols: tuple[sympy.Symbol, ...], f_symbolic: sympy.Matrix,
                 L: 'np.ndarray[np.floating]', x_0: tuple['np.floating', ...],
                 y_values: tuple['np.ndarray[np.floating], ...'], max_iteration=1000):

    grad_f_symbolic = f_symbolic.jacobian(x_symbols)

    f = sympy.lambdify((*x_symbols, *y_symbols), f_symbolic, modules='numpy')
    grad_f = sympy.lambdify((*x_symbols, *y_symbols), grad_f_symbolic, modules='numpy')

    P = np.eye(len(L))

    for i in range(max_iteration):
        L_0 = np.squeeze(f(*x_0, *y_values))
        l_reduced = L - L_0

        try:
            A = np.squeeze(grad_f(*x_0, *y_values)).transpose()
            N = A.transpose() @ P @ A

            dx = np.linalg.inv(N) @ A.transpose() @ l_reduced
            x_i = (*(np.array(x_0) + dx),)
            # print(f"Step {i:3d}\t\t{x_i}")
            x_0 = x_i

            if np.linalg.norm(dx) < 0.0001:
                print(f"Final parameters after step {i:3d}\t\t{x_i}")
                return x_0

        except np.linalg.LinAlgError as err:
            if 'Singular matrix' in str(err):
                logger = logging.getLogger(__name__)
                logger.warning(f"Iteration {i} produces singular N matrix.")
                break
            else:
                raise
        except:
            raise

    if i >= max_iteration - 1:
        logger = logging.getLogger(__name__)
        logger.warning(f"Doesn't converge.")
    return (np.nan,) * len(x_symbols)
