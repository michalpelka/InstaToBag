"""The HDMapping calibration reader, against the real files that motivated it."""

import copy
import json
import math

import pytest

from insta360_to_bag import hdmapping
from insta360_to_bag.calibration import SOURCE_METADATA, parse_offset_v2

from conftest import X5_OFFSET_V2

# Verbatim from HDMapping's calibration of an Insta360 X5 against an Ouster lidar.
X5_CAM_FRONT = {
    "extrinsics": {
        "T_lidar_to_camera_4x4": [
            [-0.02286391146481037, -0.009434384293854237, -0.9996940493583679, 0.0],
            [-0.9988134503364563, 0.04322366416454315, 0.022435856983065605, -0.0],
            [0.042998772114515305, 0.9990208745002747, -0.010411452502012253, -0.0],
            [0, 0, 0, 1],
        ],
        "camera_position_in_world_xyz": [0.0, 0.0, 0.0],
        "camera_rotation_matrix_in_world": [
            [-0.02286391146481037, -0.9988134503364563, 0.042998772114515305],
            [-0.009434384293854237, 0.04322366416454315, 0.9990208745002747],
            [-0.9996940493583679, 0.022435856983065605, -0.010411452502012253],
        ],
    },
    "serial": "IAHEA2503XUXRF",
    "model": "Insta360 X5",
    "firmware": "v1.11.10_build1",
    "intrinsics": {
        "cx": 1926.44287109375,
        "cy": 1915.5714111328125,
        "fx": 3071.17138671875,
        "fy": 3070.764404296875,
        "height": 3840,
        "k1": 0.18967017531394958,
        "k2": 2.0661227703094482,
        "k3": -3.315551280975342,
        "k4": 0.0,
        "k5": 0.0,
        "k6": 0.0,
        "model": "mei",
        "p1": 0.0005239499732851982,
        "p2": 3.870000000461005e-05,
        "width": 3840,
        "xi": 2.0,
    },
}

# The back lens of the same rig, which is the one that sits away from the lidar origin.
X5_CAM_BACK_EXTRINSICS = {
    "T_lidar_to_camera_4x4": [
        [0.016286486759781837, 0.011298257857561111, 0.9998035430908203,
         0.12297992408275604],
        [-0.9992468953132629, 0.03540508821606636, 0.01587732508778572,
         0.1726779341697693],
        [-0.035218745470047, -0.9993091821670532, 0.0118663739413023,
         -0.035225752741098404],
        [0, 0, 0, 1],
    ],
    "camera_position_in_world_xyz": [
        0.16930438578128815, -0.04270455241203308, -0.12527942657470703,
    ],
    "camera_rotation_matrix_in_world": [
        [0.016286486759781837, -0.9992468953132629, -0.035218745470047],
        [0.011298257857561111, 0.03540508821606636, -0.9993091821670532],
        [0.9998035430908203, 0.01587732508778572, 0.0118663739413023],
    ],
}


@pytest.fixture
def write_calibration(tmp_path):
    def _write(document, name="cam_front.json"):
        path = tmp_path / name
        path.write_text(json.dumps(document))
        return str(path)

    return _write


def _edited(**intrinsics):
    document = copy.deepcopy(X5_CAM_FRONT)
    document["intrinsics"].update(intrinsics)
    return document


# -- extrinsics -------------------------------------------------------------


def test_pose_inverts_the_lidar_to_camera_transform(write_calibration):
    document = copy.deepcopy(X5_CAM_FRONT)
    document["extrinsics"] = copy.deepcopy(X5_CAM_BACK_EXTRINSICS)
    pose = hdmapping.load(write_calibration(document)).pose
    # T takes lidar points into the camera; the transform ROS publishes is its inverse,
    # so the translation is -R^T t, not the t in the file.
    assert pose.translation == pytest.approx(
        X5_CAM_BACK_EXTRINSICS["camera_position_in_world_xyz"], abs=1e-7
    )


def test_pose_rotation_matches_the_file_and_is_a_unit_quaternion(write_calibration):
    pose = hdmapping.load(write_calibration(X5_CAM_FRONT)).pose
    assert math.fsum(c * c for c in pose.rotation) == pytest.approx(1.0, abs=1e-9)
    # Rotating the optical axes by it must reproduce camera_rotation_matrix_in_world,
    # whose columns are the camera's axes expressed in the lidar frame.
    stated = X5_CAM_FRONT["extrinsics"]["camera_rotation_matrix_in_world"]
    for axis, column in enumerate(range(3)):
        unit = [1.0 if i == axis else 0.0 for i in range(3)]
        assert _rotate(pose.rotation, unit) == pytest.approx(
            [row[column] for row in stated], abs=1e-6
        )


def _rotate(quaternion, vector):
    qx, qy, qz, qw = quaternion
    u = (qx, qy, qz)

    def cross(a, b):
        return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0]]

    t = [2.0 * c for c in cross(u, vector)]
    ut = cross(u, t)
    return [vector[i] + qw * t[i] + ut[i] for i in range(3)]


def test_the_optical_frame_reading_agrees_with_the_nominal_mount(write_calibration):
    """A calibration of a rig is a few degrees off the mount this tool assumes for it.

    This is what establishes that the file's camera frame is the ROS optical one: on the
    side-mounted rig the tool documents, cam_front's nominal rotation is
    (-0.5, 0.5, 0.5, 0.5).  A different axis convention would be tens of degrees out,
    not the couple that a real mount tolerance produces.
    """
    pose = hdmapping.load(write_calibration(X5_CAM_FRONT)).pose
    dot = abs(sum(a * b for a, b in zip(pose.rotation, (-0.5, 0.5, 0.5, 0.5))))
    assert math.degrees(2 * math.acos(min(1.0, dot))) < 5.0


def test_a_rotation_that_is_not_one_is_refused(write_calibration):
    document = copy.deepcopy(X5_CAM_FRONT)
    document["extrinsics"]["T_lidar_to_camera_4x4"][0][0] = 0.5
    with pytest.raises(hdmapping.CalibrationError, match="not a rotation"):
        hdmapping.load(write_calibration(document))


def test_a_mirroring_matrix_is_refused(write_calibration):
    document = copy.deepcopy(X5_CAM_FRONT)
    # Flipping one row keeps it orthonormal but makes the determinant negative.
    document["extrinsics"]["T_lidar_to_camera_4x4"][0] = [
        -v for v in document["extrinsics"]["T_lidar_to_camera_4x4"][0]
    ]
    del document["extrinsics"]["camera_rotation_matrix_in_world"]
    with pytest.raises(hdmapping.CalibrationError, match="negative determinant"):
        hdmapping.load(write_calibration(document))


def test_small_rounding_in_the_rotation_is_accepted_with_a_warning(write_calibration):
    document = copy.deepcopy(X5_CAM_FRONT)
    document["extrinsics"]["T_lidar_to_camera_4x4"][0][0] += 1e-3
    calibration = hdmapping.load(write_calibration(document))
    assert any("orthonormal" in problem for problem in calibration.warnings)


def test_a_redundant_field_that_disagrees_is_reported(write_calibration):
    document = copy.deepcopy(X5_CAM_FRONT)
    document["extrinsics"]["camera_position_in_world_xyz"] = [1.0, 0.0, 0.0]
    calibration = hdmapping.load(write_calibration(document))
    assert any(
        "camera_position_in_world_xyz" in problem for problem in calibration.warnings
    )
    # The 4x4 still wins, so the bag never gets the disputed value.
    assert calibration.pose.translation == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)


def test_a_disagreeing_rotation_matrix_is_reported(write_calibration):
    document = copy.deepcopy(X5_CAM_FRONT)
    document["extrinsics"]["camera_rotation_matrix_in_world"][0][0] = 0.9
    calibration = hdmapping.load(write_calibration(document))
    assert any(
        "camera_rotation_matrix_in_world" in problem for problem in calibration.warnings
    )


def test_the_redundant_fields_are_optional(write_calibration):
    document = copy.deepcopy(X5_CAM_FRONT)
    del document["extrinsics"]["camera_position_in_world_xyz"]
    del document["extrinsics"]["camera_rotation_matrix_in_world"]
    assert hdmapping.load(write_calibration(document)).warnings == []


# -- intrinsics -------------------------------------------------------------


def test_identification_is_read_and_the_name_comes_from_the_file(write_calibration):
    calibration = hdmapping.load(write_calibration(X5_CAM_FRONT, "cam_front.json"))
    assert calibration.serial == "IAHEA2503XUXRF"
    assert calibration.model == "Insta360 X5"
    assert calibration.firmware == "v1.11.10_build1"
    assert calibration.name == "cam_front"


def test_intrinsics_are_read_on_their_own_calibrated_size(write_calibration):
    calibration = hdmapping.load(write_calibration(X5_CAM_FRONT))
    assert (calibration.width, calibration.height) == (3840, 3840)
    assert calibration.xi == 2.0
    assert calibration.fx == pytest.approx(3071.17138671875)
    assert calibration.distortion == pytest.approx(
        [0.18967017531394958, 2.0661227703094482, -3.315551280975342,
         0.0005239499732851982, 3.870000000461005e-05]
    )


def test_a_lens_rescales_onto_the_frame_and_records_where_it_came_from(write_calibration):
    path = write_calibration(X5_CAM_FRONT)
    lens = hdmapping.load(path).lens(0, 1920, 1920)
    assert lens.canvas == (3840, 3840)
    assert lens.scale == pytest.approx((0.5, 0.5))
    assert lens.fx == pytest.approx(3071.17138671875 / 2)
    assert lens.cx == pytest.approx(1926.44287109375 / 2)
    assert lens.source == f"hdmapping {path}"
    assert lens.source != SOURCE_METADATA


def test_the_fit_is_the_same_one_the_camera_metadata_gets(write_calibration):
    """These intrinsics are the camera's own model, so the equidistant fit must behave
    exactly as it does for the metadata calibration -- same size of error, same reach."""
    lens = hdmapping.load(write_calibration(X5_CAM_FRONT)).lens(0, 2880, 2880)
    reference = parse_offset_v2(X5_OFFSET_V2, 2880, 2880)[0]
    assert lens.fx == pytest.approx(reference.fx, rel=1e-3)
    assert lens.cx == pytest.approx(reference.cx, rel=1e-3)
    assert lens.equidistant == pytest.approx(reference.equidistant, rel=1e-3)
    assert lens.fit_max_angle_deg == pytest.approx(reference.fit_max_angle_deg, rel=1e-2)


def test_a_model_that_is_not_mei_is_refused(write_calibration):
    with pytest.raises(hdmapping.CalibrationError, match="only 'mei'"):
        hdmapping.load(write_calibration(_edited(model="plumb_bob")))


@pytest.mark.parametrize("key", ["k4", "k5", "k6"])
def test_radial_terms_with_nowhere_to_go_are_reported(write_calibration, key):
    calibration = hdmapping.load(write_calibration(_edited(**{key: 0.25})))
    assert any(
        key in problem and "dropped" in problem for problem in calibration.warnings
    )


@pytest.mark.parametrize(
    "intrinsics",
    [
        {"fx": 0.0},
        {"fy": -1.0},
        {"xi": -1.0},
        {"width": 0},
        {"height": -3840},
    ],
)
def test_unusable_intrinsics_are_refused(write_calibration, intrinsics):
    with pytest.raises(hdmapping.CalibrationError):
        hdmapping.load(write_calibration(_edited(**intrinsics)))


@pytest.mark.parametrize("key", ["fx", "cx", "xi", "width"])
def test_a_missing_intrinsic_is_refused(write_calibration, key):
    document = copy.deepcopy(X5_CAM_FRONT)
    del document["intrinsics"][key]
    with pytest.raises(hdmapping.CalibrationError, match=key):
        hdmapping.load(write_calibration(document))


def test_absent_tangential_terms_default_to_zero(write_calibration):
    document = copy.deepcopy(X5_CAM_FRONT)
    del document["intrinsics"]["p1"]
    del document["intrinsics"]["p2"]
    assert hdmapping.load(write_calibration(document)).distortion[3:] == [0.0, 0.0]


# -- whole-file failures ----------------------------------------------------


def test_a_missing_file_is_refused(tmp_path):
    with pytest.raises(hdmapping.CalibrationError, match="could not read"):
        hdmapping.load(str(tmp_path / "nope.json"))


def test_a_file_that_is_not_json_is_refused(tmp_path):
    path = tmp_path / "cam_front.json"
    path.write_text("{not json")
    with pytest.raises(hdmapping.CalibrationError, match="not valid JSON"):
        hdmapping.load(str(path))


@pytest.mark.parametrize(
    "document",
    [
        [],
        {},
        {"extrinsics": {}, "intrinsics": X5_CAM_FRONT["intrinsics"]},
        {"extrinsics": X5_CAM_FRONT["extrinsics"]},
        {"extrinsics": {"T_lidar_to_camera_4x4": [[1, 0, 0, 0]]},
         "intrinsics": X5_CAM_FRONT["intrinsics"]},
        {"extrinsics": {"T_lidar_to_camera_4x4": "identity"},
         "intrinsics": X5_CAM_FRONT["intrinsics"]},
    ],
)
def test_structurally_wrong_files_are_refused(write_calibration, document):
    with pytest.raises(hdmapping.CalibrationError):
        hdmapping.load(write_calibration(document))


def test_the_error_names_the_file_it_came_from(write_calibration):
    path = write_calibration(_edited(model="omni"))
    with pytest.raises(hdmapping.CalibrationError, match=path):
        hdmapping.load(path)
