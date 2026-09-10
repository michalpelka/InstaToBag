import math

import pytest
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from mcap_ros2.writer import Writer

from insta360_to_bag import msgdefs
from insta360_to_bag.convert import (
    BODY_TO_OPTICAL,
    DEFAULT_CAMERA_RPY,
    LENS_NAMES,
    TF_STATIC_TOPIC,
    _register_latched_channel,
    static_transforms,
)


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _rotate(quaternion, vector):
    """Express a child-frame axis in the parent frame: v' = v + 2w(u x v) + 2u x (u x v)."""
    qx, qy, qz, qw = quaternion
    u = (qx, qy, qz)
    t = tuple(2.0 * c for c in _cross(u, vector))
    ut = _cross(u, t)
    return tuple(vector[i] + qw * t[i] + ut[i] for i in range(3))


def _lens_axes(camera_rpy):
    """Each lens's viewing direction, image right and image down in the lidar frame."""
    transforms = static_transforms(
        [("cam_front", "front"), ("cam_back", "back")], 0, "lidar", (0.0, 0.0, 0.0), camera_rpy
    )
    axes = {}
    for transform in transforms:
        r = transform["transform"]["rotation"]
        q = (r["x"], r["y"], r["z"], r["w"])
        axes[transform["child_frame_id"]] = {
            "view": _rotate(q, (0.0, 0.0, 1.0)),
            "right": _rotate(q, (1.0, 0.0, 0.0)),
            "down": _rotate(q, (0.0, 1.0, 0.0)),
        }
    return axes


def _near(vector):
    return pytest.approx(vector, abs=1e-12)


def test_every_lens_has_a_body_rotation():
    assert set(BODY_TO_OPTICAL) == set(LENS_NAMES)


@pytest.mark.parametrize("name", sorted(BODY_TO_OPTICAL))
def test_rotations_are_unit_quaternions(name):
    assert math.fsum(c * c for c in BODY_TO_OPTICAL[name]) == pytest.approx(1.0)


def test_an_upright_camera_with_no_rotation_looks_along_the_body_axes():
    axes = _lens_axes((0.0, 0.0, 0.0))
    assert axes["front"] == {"view": _near((1, 0, 0)), "right": _near((0, -1, 0)),
                             "down": _near((0, 0, -1))}
    assert axes["back"] == {"view": _near((-1, 0, 0)), "right": _near((0, 1, 0)),
                            "down": _near((0, 0, -1))}


def test_yaw_turns_an_upright_camera_to_look_sideways():
    axes = _lens_axes((0.0, 0.0, -90.0))
    assert axes["front"]["view"] == _near((0, -1, 0))
    assert axes["back"]["view"] == _near((0, 1, 0))
    assert axes["front"]["down"] == _near((0, 0, -1)) == axes["back"]["down"]


def test_default_mount_is_the_side_mounted_rig():
    # On its side with the lens end toward the lidar's +x: world-up is image -x on the
    # front lens and image +x on the back one, and the rig (lidar-backward) sits at the
    # bottom of both images -- as seen in that rig's own frames.
    axes = _lens_axes(DEFAULT_CAMERA_RPY)
    assert axes["front"] == {"view": _near((0, 1, 0)), "right": _near((0, 0, -1)),
                             "down": _near((-1, 0, 0))}
    assert axes["back"] == {"view": _near((0, -1, 0)), "right": _near((0, 0, 1)),
                            "down": _near((-1, 0, 0))}


def test_static_transforms_place_both_lenses_at_the_camera_position():
    transforms = static_transforms(
        [("cam_front", "front_optical"), ("cam_back", "back_optical")],
        1_500_000_123,
        "os_sensor",
        (0.1, -0.2, 0.3),
        (0.0, 0.0, 0.0),
    )
    assert [t["child_frame_id"] for t in transforms] == ["front_optical", "back_optical"]
    for transform in transforms:
        assert transform["header"] == {
            "stamp": {"sec": 1, "nanosec": 500_000_123},
            "frame_id": "os_sensor",
        }
        assert transform["transform"]["translation"] == {"x": 0.1, "y": -0.2, "z": 0.3}
    # With no mount rotation, the lidar and body frames coincide.
    rotation = transforms[1]["transform"]["rotation"]
    assert (rotation["x"], rotation["y"], rotation["z"], rotation["w"]) == pytest.approx(
        BODY_TO_OPTICAL["cam_back"]
    )


def test_tf_static_round_trips_through_a_bag_with_latched_qos(tmp_path):
    path = tmp_path / "tf.mcap"
    with open(path, "wb") as handle:
        writer = Writer(handle)
        schema = writer.register_msgdef(
            msgdefs.TF_MESSAGE, msgdefs.DEFINITIONS[msgdefs.TF_MESSAGE]
        )
        _register_latched_channel(writer, TF_STATIC_TOPIC, schema)
        writer.write_message(
            TF_STATIC_TOPIC,
            schema,
            {"transforms": static_transforms(
                [("cam_back", "insta360_cam_back_optical_frame")],
                7, "lidar", (1.0, 2.0, 3.0), DEFAULT_CAMERA_RPY,
            )},
            log_time=7,
        )
        writer.finish()

    with open(path, "rb") as handle:
        reader = make_reader(handle, decoder_factories=[DecoderFactory()])
        channels = list(reader.get_summary().channels.values())
        decoded = [message for _, _, _, message in reader.iter_decoded_messages()]
    assert [channel.topic for channel in channels] == [TF_STATIC_TOPIC]
    # durability 1 is RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL.
    assert "durability: 1" in channels[0].metadata["offered_qos_profiles"]

    (message,) = decoded
    (transform,) = message.transforms
    assert transform.header.frame_id == "lidar"
    assert transform.child_frame_id == "insta360_cam_back_optical_frame"
    assert (transform.transform.translation.x, transform.transform.translation.y,
            transform.transform.translation.z) == (1.0, 2.0, 3.0)
