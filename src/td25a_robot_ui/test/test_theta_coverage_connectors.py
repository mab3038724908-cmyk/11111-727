import math

import numpy as np

from td25a_robot_ui.algorithms import grid_coverage
from td25a_robot_ui.algorithms.grid_coverage import (
    _astar,
    _connect_points,
    _polyline_is_free,
    _supercover_line_cells,
    _theta_star,
)


def _connector_mask():
    free = np.ones((24, 32), dtype=bool)
    # A tall island forces the connector to choose its top or bottom side.
    free[6:19, 14:18] = False
    return free


def _cell_path_length(path):
    return sum(
        math.hypot(b[0] - a[0], b[1] - a[1])
        for a, b in zip(path, path[1:])
    )


def test_theta_star_is_any_angle_and_never_leaves_coverage_mask():
    free = _connector_mask()
    start = (12, 3)
    goal = (12, 28)

    theta = _theta_star(free, start, goal)
    astar = _astar(free, start, goal)

    assert theta[0] == start
    assert theta[-1] == goal
    assert len(theta) < len(astar)
    assert _cell_path_length(theta) <= _cell_path_length(astar) + 1.0
    for a, b in zip(theta, theta[1:]):
        assert all(free[y, x] for y, x in _supercover_line_cells(a, b))


def test_theta_star_cannot_squeeze_through_an_exact_blocked_corner():
    free = np.ones((3, 3), dtype=bool)
    free[0, 1] = False
    free[1, 0] = False

    assert _theta_star(free, (0, 0), (1, 1)) == []


def test_offline_theta_connector_restores_dense_world_path():
    free = _connector_mask()
    path = _connect_points(
        free,
        start=(0.35, 1.25),
        goal=(2.85, 1.25),
        resolution=0.1,
        origin_x=0.0,
        origin_y=0.0,
        path_step_m=0.1,
        prefer_theta=True,
    )

    assert path
    assert _polyline_is_free(free, path, 0.1, 0.0, 0.0)
    assert max(
        math.hypot(b[0] - a[0], b[1] - a[1])
        for a, b in zip(path, path[1:])
    ) <= 0.101


def test_theta_failure_falls_back_to_existing_astar(monkeypatch):
    free = _connector_mask()
    monkeypatch.setattr(grid_coverage, "_theta_star", lambda *args, **kwargs: [])

    path = _connect_points(
        free, (0.35, 1.25), (2.85, 1.25), 0.1, 0.0, 0.0, 0.1,
        prefer_theta=True,
    )

    assert path
    assert _polyline_is_free(free, path, 0.1, 0.0, 0.0)


def test_online_default_connector_does_not_enable_theta(monkeypatch):
    free = _connector_mask()

    def fail_if_called(*args, **kwargs):
        raise AssertionError("online/default connector unexpectedly called Theta*")

    monkeypatch.setattr(grid_coverage, "_theta_star", fail_if_called)
    path = _connect_points(
        free, (0.35, 1.25), (2.85, 1.25), 0.1, 0.0, 0.0, 0.1)

    assert path
