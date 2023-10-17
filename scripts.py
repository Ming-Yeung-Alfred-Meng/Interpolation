import numpy as np
import matplotlib.pyplot as plt


def hermite_interpolation_1d(keyframes: np.ndarray,
                             gradients: np.ndarray,
                             interpolation_count: int = 100) -> np.ndarray:
    """
    Hermite interpolation of values between keyframes.
    :param keyframes: 1D ndarray of values between which values are interpolated. It must contain at least two values.
    :param gradients: gradients w.r.t the interpolation parameter at keyframes. It must contain at least two values.
    :param interpolation_count: number of intermediate points between two keyframes (inclusive).
    :return: interpolation_count x (len(keyframes) - 1) ndarray of interpolated values.
    """
    assert len(keyframes.shape) == 1 and keyframes.shape[0] >= 2
    assert len(gradients.shape) == 1 and gradients.shape[0] >= 2

    return (np.power(np.linspace([0, 0, 0, 0],
                                 [1, 1, 1, 1],
                                 num=interpolation_count),
                     range(3, -1, -1))
            @ np.array([[2, -2, 1, 1],
                        [-3, 3, -2, -1],
                        [0, 0, 1, 0],
                        [1, 0, 0, 0]])
            @ np.stack((keyframes[:-1],
                        keyframes[1:],
                        gradients[: -1],
                        gradients[1:]), axis=0))


def catmullrom_spline_1d(keyframes: np.ndarray,
                         interpolation_count: int = 100) -> np.ndarray:
    """
    Catmull-Rom interpolation of values between keyframes.
    :param keyframes: 1D ndarray of values between which values are interpolated. It must contain at least four values.
    :param gradients: gradients w.r.t the interpolation parameter at keyframes. It must contain at least two values.
    :param interpolation_count: number of intermediate points between two keyframes (inclusive).
    :return: interpolation_count x (len(keyframes) - 1) ndarray of interpolated values.
    """
    assert len(keyframes.shape) == 1 and keyframes.shape[0] >= 4

    return (np.power(np.linspace([0, 0, 0, 0],
                                 [1, 1, 1, 1],
                                 num=interpolation_count),
                     [3, 2, 1, 0])
            @ (0.5 * np.array([[-1, 3, -3, 1],
                              [2, -5, 4, -1],
                              [-1, 0, 1, 0],
                              [0, 2, 0, 0]]))
            @ np.stack((keyframes[:-3],
                        keyframes[1:-2],
                        keyframes[2:-1],
                        keyframes[3:]), axis=0))


def plot_interpolation_2d(keyframes_x: np.ndarray,
                          keyframes_y: np.ndarray,
                          gradients_x: np.ndarray,
                          gradients_y: np.ndarray,
                          interpolated_x: np.ndarray,
                          interpolated_y: np.ndarray,
                          title: str) -> None:
    """
    Plot interpolated curves.
    :param keyframes_x: x components of 2D keyframe points.
    :param keyframes_y: y components of 2D keyframe points.
    :param gradients_x: x components of gradients at 2D keyframe points.
    :param gradients_y: y components of gradients at 2D keyframe points.
    :param interpolated_x: x components of interpolated 2D points.
    :param interpolated_y: y components of interpolated 2D points.
    :param title: plot title.
    :return: None
    """

    plt.plot(interpolated_x, interpolated_y)

    plt.quiver(keyframes_x, keyframes_y, gradients_x, gradients_y,
               angles='xy', scale_units='xy', scale=10,
               units='dots', width=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.suptitle(title)
