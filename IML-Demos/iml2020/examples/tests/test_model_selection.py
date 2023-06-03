"""Python Script Template."""
import numpy as np
from ..utilities import plot_helpers
import matplotlib.pyplot as plt
# sklearn library
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge, Lasso
from sklearn.pipeline import make_pipeline

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score


def f(x):
    return x * np.sin(x)

def plot_data():
    x_plot = np.linspace(-1, 11, 100)
    f_plot = f(x_plot)
    X_plot = x_plot[:, np.newaxis]

    return X_plot, f_plot

def data(N=20):
    x = 10 * np.random.rand(N)
    X = x[:, np.newaxis]
    y = f(x) + np.random.normal(size=(N,))

    return X, y


def test_plot_data():
    X, Y = plot_data()
    # Plot Data
    fig = plt.subplot(111)
    plot_opts = {'x_label': '$x$', 'y_label': '$y$', 'title': 'Generated Data',
                 'y_lim': [np.min(Y) - 0.5, np.max(Y) + 0.5]}
    plot_helpers.plot_data(X, Y, fig=fig, options=plot_opts)
    plt.clf()


def test_pipeline():
    folds = 5
    N = 200
    n = int(N / folds)
    fold = 1
    xraw, yraw = data(N)
    x = dict()
    y = dict()
    for i in range(folds):
        x[i] = xraw[n * i:n * (i + 1)]
        y[i] = yraw[n * i:n * (i + 1)]

    X = np.empty((0, 1))
    Y = np.empty((0, ))
    for i in range(folds):
        if i == (fold - 1):
            Xval = x[i]
            Yval = y[i]
        else:
            print(x[i].shape, y[i].shape)
            X = np.concatenate((X, x[i]))
            Y = np.concatenate((Y, y[i]))

    x_plot, f_plot = plot_data()
    degree = 10
    model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=0))
    model.fit(X, Y)

    fig = plt.subplot(111)
    lw = 2
    plt.plot(x_plot, f_plot, color='cornflowerblue', linewidth=lw, label="Ground Truth")
    y_plot = model.predict(x_plot)
    plt.plot(x_plot, y_plot, color='r', linewidth=lw, label="Degree %d" % degree)

    opts = {'marker': 'b*', 'label': 'Training Points'}
    plot_helpers.plot_data(X, Y, fig=fig, options=opts)

    plot_opts = {'x_label': '$x$', 'y_label': '$y$',
                 'y_lim': [np.min(f_plot) - 3, np.max(f_plot) + 3],
                 'legend': True, 'legend_loc': 'lower left'}
    opts = {'marker': 'mX', 'label': 'Validation Points'}
    plot_opts.update(opts)
    plot_helpers.plot_data(Xval, Yval, fig=fig, options=plot_opts)

    print("Train. Error: {:.2f}".format(
        1 / X.size * np.linalg.norm(model.predict(X) - Y, 2)))
    print("Valid. Error: {:.2f}".format(
        1 / Xval.size * np.linalg.norm(model.predict(Xval) - Yval, 2)))
    plt.clf()


def test_regularization():
    X, y, w = make_regression(n_samples=10, n_features=10, coef=True, random_state=1, bias=3.5)
    clfs = dict(ridge=Ridge(), lasso=Lasso())

    alphas = np.logspace(-6, 6, 200)
    coefs = dict(ridge=[], lasso=[])
    for a in alphas:
        for name, clf in clfs.items():
            clf.set_params(alpha=a, max_iter=10000)
            clf.fit(X, y)
            coefs[name].append(clf.coef_)

    fig = plt.figure(1)
    plt.plot(alphas, coefs['ridge'])
    plt.xscale('log')
    plt.xlabel('Lambda')
    plt.ylabel('weights')
    plt.title('Ridge coefficients as a function of the regularization')

    fig = plt.figure(2)
    plt.plot(alphas, coefs['lasso'])
    plt.xscale('log')
    plt.xlabel('lambda')
    plt.ylabel('weights')
    plt.title('Lasso coefficients as a function of the regularization')


def true_fun(X):
    return np.cos(1.5 * np.pi * X)


def test_poly_reg():
    n_samples, degree, noise = 20, 10, 1

    X = np.sort(np.random.rand(n_samples))
    y = true_fun(X) + np.random.randn(n_samples) * noise

    polynomial_features = PolynomialFeatures(degree=degree,
                                             include_bias=True)
    linear_regression = LinearRegression()
    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])
    pipeline.fit(X[:, np.newaxis], y)

    # Evaluate the models using crossvalidation
    scores = cross_val_score(pipeline, X[:, np.newaxis], y,
                             scoring="neg_mean_squared_error", cv=10)

    X_test = np.linspace(0, 1, 100)
    plt.plot(X_test, pipeline.predict(X_test[:, np.newaxis]), label="Model")
    plt.plot(X_test, true_fun(X_test), label="True function")
    plt.scatter(X, y, edgecolor='b', s=20, label="Samples")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.ylim((-2, 2))
    plt.legend(loc="best")
    plt.title("Degree {}\nMSE = {:.2e}(+/- {:.2e})".format(
        degree, -scores.mean(), scores.std()))


def test_bias_variance():
    n_samples = 300
    np.random.seed(0)
    score = []

    degrees = np.arange(1, 10, 1)
    for degree in degrees:
        X = np.sort(np.random.rand(n_samples))
        y = true_fun(X) + np.random.randn(n_samples) * 1

        polynomial_features = PolynomialFeatures(degree=degree,
                                                 include_bias=True)
        linear_regression = LinearRegression()
        pipeline = Pipeline([("polynomial_features", polynomial_features),
                             ("linear_regression", linear_regression)])
        pipeline.fit(X[:, np.newaxis], y)

        # Evaluate the models using crossvalidation
        scores = cross_val_score(pipeline, X[:, np.newaxis], y,
                                 scoring="neg_mean_squared_error", cv=10)

        score.append(-scores.mean())

    plt.plot(degrees, score)
    plt.ylabel('MSE')
    plt.xlabel('Polynomial degree (Model Complexity)')
    plt.ylim([0, 2])