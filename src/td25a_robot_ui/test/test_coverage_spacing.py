from td25a_robot_ui.algorithms.grid_coverage import coverage_swath_spacing
import pytest

def test_spacing_from_clean_width_not_safety_radius():
    assert abs(coverage_swath_spacing(0.70, 0.875) - 0.6125) < 1e-6

def test_spacing_default_overlap():
    assert abs(coverage_swath_spacing(0.70) - 0.70 * 0.875) < 1e-6

def test_spacing_rejects_nonpositive():
    with pytest.raises(ValueError):
        coverage_swath_spacing(0.0)
