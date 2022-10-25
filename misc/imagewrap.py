import numpy as np

_small = np.finfo(np.float_).eps


def _U(x):
    return x * np.where(x < _small, 0, np.log(x) / 2)


def _interpoint_distances(points):
    xd = np.subtract.outer(points[:, 0], points[:, 0])
    yd = np.subtract.outer(points[:, 1], points[:, 1])
    return np.square(xd) + np.square(yd)


def _make_L_matrix(points):
    n = len(points)
    K = _U(_interpoint_distances(points))
    P = np.ones((n, 3))
    P[:, 1:] = points
    O = np.zeros((3, 3))
    L = np.asarray(np.bmat([[K, P], [P.transpose(), O]]))
    return L
