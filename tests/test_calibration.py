import pytest

from insta360_to_bag.calibration import DISTORTION_MODEL, parse_offset_v2, summarise

# Verbatim from an Insta360 X5 capture (metadata field 54).
X5_OFFSET_V2 = (
    "2_2.000000_4299.640_4299.070_2697.020_2681.800_0.196_-0.027_89.717_"
    "0.000000_0.000000_0.000000_0.18967018_2.06612277_-3.31555128_0.00052395_"
    "0.00003870_10752_5376_113_2.000000_4273.430_4273.780_8083.250_2666.280_"
    "-0.136_-0.096_90.887_0.002375_-0.000169_-0.031969_0.19482063_1.98416793_"
    "-3.08075333_0.00059304_0.00104311_10752_5376_113_197632"
)


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


def test_distortion_and_pose_pass_through_verbatim():
    lens = parse_offset_v2(X5_OFFSET_V2, 2880, 2880)[0]
    assert lens.distortion == [
        0.18967018, 2.06612277, -3.31555128, 0.00052395, 0.00003870,
    ]
    assert lens.rotation_deg == (0.196, -0.027, 89.717)
    assert lens.translation == (0.0, 0.0, 0.0)


def test_projection_matrices_are_well_formed():
    lens = parse_offset_v2(X5_OFFSET_V2, 2880, 2880)[0]
    assert lens.k == [lens.fx, 0.0, lens.cx, 0.0, lens.fy, lens.cy, 0.0, 0.0, 1.0]
    assert lens.r == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    assert len(lens.p) == 12 and lens.p[3] == 0.0 and lens.p[10] == 1.0


def test_summary_mentions_the_source_canvas():
    text = summarise(parse_offset_v2(X5_OFFSET_V2, 2880, 2880)[0])
    assert "10752x5376" in text and "fx=" in text


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
    ],
)
def test_unparseable_strings_yield_no_calibration(text):
    # Degrading to "no CameraInfo" is the point: a firmware revision with a different
    # layout must never produce plausible-looking but wrong intrinsics.
    assert parse_offset_v2(text, 2880, 2880) == []


def test_distortion_model_is_explicitly_not_a_standard_one():
    assert DISTORTION_MODEL == "insta360_fisheye_v2"
    assert DISTORTION_MODEL not in {"plumb_bob", "equidistant", "rational_polynomial"}
