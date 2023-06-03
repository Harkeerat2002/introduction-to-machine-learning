"""Python Script Template."""
import pytest
import numpy as np
from ..utilities import plot_helpers
from ..utilities.load_data import polynomial_data
from ..utilities.regressors import LinearRegressor
from ..utilities.util import gradient_descent
import matplotlib.pyplot as plt


def linear_data():
    num_points = 100  # Number of training points.
    noise = 0.6  # Noise Level (needed for data generation).

    a_true = 3  # Slope.
    b_true = 1  # Intercept.
    w_true = np.array([a_true, b_true])

    X, Y = polynomial_data(num_points, noise, w_true)
    return X, Y


def cubic_data():
    num_points = 100  # Number of training points.
    noise = 0.6  # Noise Level (needed for data generation).

    w_true = np.array([-.5, .5, 1, -1])

    X, Y = polynomial_data(num_points, noise, w_true)
    return X, Y


@pytest.fixture(params=[linear_data(), cubic_data()])
def data(request):
    return request.param


@pytest.fixture(params=[1, 32, 100])
def batch_size(request):
    return request.param


@pytest.fixture(params=['Bold driver', 'AdaGrad', 'Annealing', None])
def heuristic(request):
    return request.param


class SinCos(object):
    def __init__(self):
        pass

    @property
    def number_samples(self):
        return 0

    def loss(self, w, *args):
        return np.sin(w[0]) * np.cos(w[1])

    def gradient(self, w, *args):
        return np.array([np.cos(w[0]) * np.cos(w[1]), -np.sin(w[0]) * np.sin(w[1])])


def test_plot_data(data):
    X, Y = data
    # Plot Data
    fig = plt.subplot(111);
    plot_opts = {'x_label': '$x$', 'y_label': '$y$', 'title': 'Generated Data',
                 'y_lim': [np.min(Y) - 0.5, np.max(Y) + 0.5]}
    plot_helpers.plot_data(X[:, -2], Y, fig=fig, options=plot_opts)
    plt.clf()


def test_best_response(data):
    X, Y = data
    dim = X.shape[1]
    reg = 0  # The regularizer is set to zero by now
    w_hat_closed_form = np.dot(np.linalg.pinv(np.dot(X.T, X) + reg * np.eye(dim)),
                               np.dot(X.T, Y))
    fig = plt.subplot(111)
    plot_opts = {'x_label': '$x$', 'y_label': '$y$', 'title': 'Closed Form Solution',
                 'legend': True,
                 'y_lim': [np.min(Y) - 0.5, np.max(Y) + 0.5]}

    plot_helpers.plot_data(X[:, -2], Y, fig=fig, options=plot_opts)
    plot_helpers.plot_fit(X, w_hat_closed_form, fig=fig, options=plot_opts)
    plt.clf()


# def test_gd(data, batch_size, heuristic):
def test_gd(data):
    X, Y = data
    eta0 = 0.1
    n_iter = 2
    batch_size=1
    heuristic=None

    regressor = LinearRegressor(X, Y)
    w0 = np.zeros(X.shape[1])

    opts = {'eta0': eta0, 'n_iter': n_iter, 'batch_size': batch_size,
            'n_samples': X.shape[0], 'algorithm': 'SGD',
            'learning_rate_scheduling': heuristic}
    trajectory, indexes = gradient_descent(w0, regressor, opts=opts)

    data_opts = {'x_label': '$x$', 'y_label': '$y$', 'title': 'Regression trajectory',
                 'legend': False,
                  'y_lim': [np.min(Y)-0.5, np.max(Y)+0.5], 'sgd_point': True}
    plot_opts = {'data_opts': data_opts}

    if X.shape[1] == 2:
        contourplot = plt.subplot(121)
        dataplot = plt.subplot(122)
        contour_opts = {'x_label': '$w_0$', 'y_label': '$w_1$',
                        'title': 'Weight trajectory', 'legend': False,
                        }
        plot_opts.update(contour_opts=contour_opts)
    else:
        contourplot = None
        dataplot = plt.subplot(111)

    plot_helpers.linear_regression_progression(X, Y, trajectory, indexes,
                                               regressor.test_loss, contourplot,
                                               dataplot, options=plot_opts)
    plt.clf()


def test_sincos(heuristic):
    loss_function = SinCos()
    w0 = np.array([0.1, 0.2])

    opts = {'eta0': 1e-1,
            'n_iter': 2,
            'learning_rate_scheduling': heuristic
            }
    trajectory, _ = gradient_descent(w0, loss_function, opts=opts)

    contourplot = plt.subplot(111)
    dataplot = None
    contour_opts = {'x_label': '$w_0$', 'y_label': '$w_1$',
                    'title': 'Weight trajectory', 'legend': False}
    plot_opts = {'contour_opts': contour_opts}

    plot_helpers.linear_regression_progression(np.array([]), np.array([]), trajectory,
                                               np.array([]),
                                               loss_function.loss,
                                               contourplot, dataplot, options=plot_opts)
    plt.clf()
