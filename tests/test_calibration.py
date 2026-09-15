import math

import pytest

from insta360_to_bag.calibration import DISTORTION_MODEL, parse_offset_v2, summarise

from conftest import X5_OFFSET_V2


def test_parses_both_lenses():
    lenses = parse_offset_v2(X5_OFFSET_V2, 2880, 2880)
    assert [lens.index for lens in lenses] == [0, 1]
    assert all(lens.canvas == (10752, 5376) for lens in lenses)


def test_principal_point_lands_near_the_frame_centre():
    # The camera expresses both lenses on one 10752-wide stitched canvas, so the
    # second lens' cx is offset by half the canvas width and has to be brought back.
    for lens in parse_offset_v2(X5_OFFSET_V2, 2880, 2880):
        assert lens.cx == pytest.approx(1440, abs=15)
        assert lens.cy == pytest.approx(1440, abs=15)
        assert lens.fx == pytest.approx(2300, abs=25)
        assert lens.fy == pytest.approx(lens.fx, rel=1e-3)


def test_intrinsics_scale_with_the_requested_frame_size():
    full = parse_offset_v2(X5_OFFSET_V2, 2880, 2880)[0]
    half = parse_offset_v2(X5_OFFSET_V2, 1440, 1440)[0]
    assert half.fx == pytest.approx(full.fx / 2)
    assert half.fy == pytest.approx(full.fy / 2)
    assert half.cx == pytest.approx(full.cx / 2)
    assert half.cy == pytest.approx(full.cy / 2)


def test_non_square_output_scales_each_axis_independently():
    lens = parse_offset_v2(X5_OFFSET_V2, 2880, 1440)[0]
    reference = parse_offset_v2(X5_OFFSET_V2, 2880, 2880)[0]
    assert lens.fx == pytest.approx(reference.fx)
    assert lens.fy == pytest.approx(reference.fy / 2)
    assert lens.cy == pytest.approx(reference.cy / 2)


def test_model_parameters_and_pose_pass_through_verbatim():
    lens = parse_offset_v2(X5_OFFSET_V2, 2880, 2880)[0]
    assert lens.xi == 2.0
    assert lens.distortion == [
        0.18967018, 2.06612277, -3.31555128, 0.00052395, 0.00003870,
    ]
    assert lens.rotation_deg == (0.196, -0.027, 89.717)
    assert lens.translation == (0.0, 0.0, 0.0)


def test_projection_matrices_use_the_equidistant_focal_length():
    lens = parse_offset_v2(X5_OFFSET_V2, 2880, 2880)[0]
    assert lens.equidistant_fx == pytest.approx(lens.fx / 3)
    assert lens.equidistant_fy == pytest.approx(lens.fy / 3)
    assert lens.k == [
        lens.equidistant_fx, 0.0, lens.cx, 0.0, lens.equidistant_fy, lens.cy, 0.0, 0.0, 1.0,
    ]
    assert lens.r == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    assert lens.p == [
        lens.equidistant_fx, 0.0, lens.cx, 0.0,
        0.0, lens.equidistant_fy, lens.cy, 0.0,
        0.0, 0.0, 1.0, 0.0,
    ]


def _unified_radius_px(lens, theta):
    """Independent restatement of the unified (Mei) model, radial part: the ray goes on
    the unit sphere, is projected from xi behind its centre, then gets k1..k3."""
    k1, k2, k3 = lens.distortion[:3]
    rho = math.sin(theta) / (math.cos(theta) + lens.xi)
    r2 = rho**2
    return lens.fx * rho * (1 + k1 * r2 + k2 * r2**2 + k3 * r2**3)


def _equidistant_radius_px(lens, theta):
    """Kannala-Brandt as cv::fisheye defines it."""
    k1, k2, k3, k4 = lens.equidistant
    theta_d = theta * (1 + k1 * theta**2 + k2 * theta**4 + k3 * theta**6 + k4 * theta**8)
    return lens.equidistant_fx * theta_d


@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize("degrees", [5, 30, 60, 90, 97, 100])
def test_equidistant_fit_reproduces_the_unified_model(index, degrees):
    lens = parse_offset_v2(X5_OFFSET_V2, 3840, 3840)[index]
    theta = math.radians(degrees)
    assert _equidistant_radius_px(lens, theta) == pytest.approx(
        _unified_radius_px(lens, theta), abs=0.5
    )


def test_fit_covers_the_image_circle_and_bounds_its_error():
    for lens in parse_offset_v2(X5_OFFSET_V2, 3840, 3840):
        # A ~200 deg lens: its image circle meets the frame edge just past 100 deg.
        assert 100 < lens.fit_max_angle_deg < 105
        # Dominated by the dropped tangential terms; a few pixels at the rim.
        assert 0 < lens.fit_error_px < 5


def test_fit_is_independent_of_the_frame_size():
    full = parse_offset_v2(X5_OFFSET_V2, 3840, 3840)[1]
    quarter = parse_offset_v2(X5_OFFSET_V2, 960, 960)[1]
    assert quarter.equidistant == pytest.approx(full.equidistant)
    assert quarter.fit_max_angle_deg == pytest.approx(full.fit_max_angle_deg)
    assert quarter.fit_error_px == pytest.approx(full.fit_error_px / 4, rel=0.01)


def test_summary_mentions_the_model_and_source_canvas():
    text = summarise(parse_offset_v2(X5_OFFSET_V2, 2880, 2880)[0])
    assert "10752x5376" in text and "fx=" in text and "xi=2" in text


@pytest.mark.parametrize(
    "text",
    [
        "",
        "2",
        "not_a_number_at_all",
        "2_2.000000_4299.640",  # truncated before the first lens group ends
        "3_" + "_".join(["1"] * 19),  # claims three lenses, carries one
        "2_" + "_".join(["nope"] * 39),
        "1_" + "_".join(["1"] * 16 + ["0", "0", "113"]),  # zero canvas
        "1_-1_" + "_".join(["1"] * 15 + ["10752", "5376", "113"]),  # xi = -1
    ],
)
def test_unparseable_strings_yield_no_calibration(text):
    # Degrading to "no CameraInfo" is the point: a firmware revision with a different
    # layout must never produce plausible-looking but wrong intrinsics.
    assert parse_offset_v2(text, 2880, 2880) == []


def test_distortion_model_is_one_ros_defines():
    # sensor_msgs/distortion_models.hpp
    assert DISTORTION_MODEL in {"plumb_bob", "rational_polynomial", "equidistant"}
    assert DISTORTION_MODEL == "equidistant"
