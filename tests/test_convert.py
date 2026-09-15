import json
import math
import shutil
import struct

import pytest

from mcap.reader import make_reader
from mcap_ros2.writer import Writer as McapRos2Writer

from insta360_to_bag import hdmapping, metadata as metadata_mod
from insta360_to_bag.convert import (
    CAMERA_METADATA_RECORD,
    TF_STATIC_TOPIC,
    Options,
    Summary,
    _Camera,
    _external_calibration,
    _metadata_payload,
    _write_camera_metadata,
    convert,
    resolve_scale,
    static_transforms,
)

from conftest import REAL_CAPTURE, requires_real_capture
from test_hdmapping import X5_CAM_FRONT

requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not on PATH",
)


# -- scale resolution -------------------------------------------------------


def test_no_scale_keeps_the_source_size():
    assert resolve_scale(None, 2880, 2880) == (2880, 2880)
    assert resolve_scale("", 2880, 2880) == (2880, 2880)


def test_explicit_dimensions():
    assert resolve_scale("1440x720", 2880, 2880) == (1440, 720)
    assert resolve_scale(" 1440X720 ", 2880, 2880) == (1440, 720)


def test_factor_scaling_rounds_to_even_dimensions():
    assert resolve_scale("0.5", 2880, 2880) == (1440, 1440)
    # 2880 * 0.3333 = 959.9, which must not become an odd number.
    width, height = resolve_scale("0.3333", 2880, 2880)
    assert width % 2 == 0 and height % 2 == 0


def test_factor_of_one_is_a_no_op():
    assert resolve_scale("1.0", 2880, 2880) == (2880, 2880)


@pytest.mark.parametrize("spec", ["0", "-0.5", "1.5", "2", "abc", "x", "0x0", "-4x8"])
def test_bad_scale_specs_are_rejected(spec):
    with pytest.raises(ValueError):
        resolve_scale(spec, 2880, 2880)


# -- option validation ------------------------------------------------------


def test_unknown_compression_is_rejected(tmp_path):
    options = Options(
        input_path=REAL_CAPTURE,
        output_path=str(tmp_path / "out.mcap"),
        compression="brotli",
    )
    with pytest.raises(ValueError, match="unknown compression"):
        convert(options)


def test_a_file_without_a_trailer_is_rejected(tmp_path):
    source = tmp_path / "plain.mp4"
    source.write_bytes(b"\x00" * 4096)
    options = Options(input_path=str(source), output_path=str(tmp_path / "out.mcap"))
    with pytest.raises(Exception):
        convert(options)


# -- end to end -------------------------------------------------------------


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    if not shutil.which("ffmpeg") or not __import__("os").path.exists(REAL_CAPTURE):
        pytest.skip("needs ffmpeg and the sample capture")
    output = tmp_path_factory.mktemp("bag") / "out.mcap"
    summary = convert(
        Options(
            input_path=REAL_CAPTURE,
            output_path=str(output),
            max_frames=4,
            jpeg_quality=12,
        )
    )
    return summary, str(output)


@requires_ffmpeg
@requires_real_capture
def test_every_requested_stream_is_present(converted):
    summary, _ = converted
    counts = summary.message_counts
    assert counts["/insta360/cam_front/image/compressed"] == 4
    assert counts["/insta360/cam_back/image/compressed"] == 4
    assert counts["/insta360/cam_front/camera_info"] == 4
    assert counts["/insta360/cam_back/camera_info"] == 4
    assert counts["/insta360/exposure_time"] == 4
    assert counts["/insta360/imu"] == 16752
    assert counts["/insta360/metadata"] == 1
    assert counts["/insta360/preview/image"] == 1
    assert counts["/insta360/audio"] >= 1
    assert counts["/tf_static"] == 1
    assert summary.warnings == []


@requires_ffmpeg
@requires_real_capture
def test_messages_are_written_in_time_order_with_matching_header_stamps(converted):
    from mcap_ros2.reader import read_ros2_messages

    _, path = converted
    previous = -1
    for message in read_ros2_messages(path):
        assert message.log_time_ns >= previous
        previous = message.log_time_ns
        header = getattr(message.ros_msg, "header", None)
        if header is not None:
            stamped = header.stamp.sec * 10**9 + header.stamp.nanosec
            assert stamped == message.log_time_ns


@requires_ffmpeg
@requires_real_capture
def test_schemas_are_embedded_so_the_bag_is_self_describing(converted):
    from mcap.reader import make_reader

    _, path = converted
    with open(path, "rb") as handle:
        summary = make_reader(handle).get_summary()
    by_topic = {
        channel.topic: summary.schemas[channel.schema_id]
        for channel in summary.channels.values()
    }
    assert by_topic["/insta360/imu"].name == "sensor_msgs/msg/Imu"
    assert by_topic["/insta360/cam_front/image/compressed"].name == (
        "sensor_msgs/msg/CompressedImage"
    )
    for schema in by_topic.values():
        assert schema.encoding == "ros2msg"
        assert schema.data, "schema definition must travel inside the bag"


@requires_ffmpeg
@requires_real_capture
def test_imu_is_physically_plausible(converted):
    from insta360_to_bag.sensors import STANDARD_GRAVITY
    from mcap_ros2.reader import read_ros2_messages

    _, path = converted
    magnitudes = []
    for message in read_ros2_messages(path, topics=["/insta360/imu"]):
        accel = message.ros_msg.linear_acceleration
        magnitudes.append(math.sqrt(accel.x**2 + accel.y**2 + accel.z**2))
        assert message.ros_msg.orientation_covariance[0] == -1.0
        assert message.ros_msg.header.frame_id == "insta360_imu"
    mean_g = sum(magnitudes) / len(magnitudes) / STANDARD_GRAVITY
    # Gravity dominates a handheld clip, so the mean magnitude must sit near 1 g.
    # This is what pins the channel order and the offset-binary encoding.
    assert mean_g == pytest.approx(1.0, abs=0.05)


@requires_ffmpeg
@requires_real_capture
def test_images_are_decodable_jpeg_of_the_expected_size(converted):
    from mcap_ros2.reader import read_ros2_messages

    _, path = converted
    seen = 0
    for message in read_ros2_messages(
        path, topics=["/insta360/cam_front/image/compressed"]
    ):
        data = bytes(message.ros_msg.data)
        assert message.ros_msg.format == "jpeg"
        assert data.startswith(b"\xff\xd8") and data.endswith(b"\xff\xd9")
        index = data.index(b"\xff\xc0")
        height, width = struct.unpack(">HH", data[index + 5 : index + 9])
        assert (width, height) == (2880, 2880)
        seen += 1
    assert seen == 4


@requires_ffmpeg
@requires_real_capture
def test_camera_info_matches_the_images_and_carries_an_equidistant_model(converted):
    from mcap_ros2.reader import read_ros2_messages

    _, path = converted
    for message in read_ros2_messages(path, topics=["/insta360/cam_front/camera_info"]):
        info = message.ros_msg
        assert (info.width, info.height) == (2880, 2880)
        assert info.distortion_model == "equidistant"
        assert len(info.d) == 4 and len(info.k) == 9 and len(info.p) == 12
        # Equidistant focal length is the unified one over (1 + xi), xi = 2 on an X5.
        assert info.k[0] == pytest.approx(2300 / 3, abs=10)
        assert info.k[2] == pytest.approx(1440, abs=15)  # cx
        assert info.k[5] == pytest.approx(1440, abs=15)  # cy
        assert info.header.frame_id == "insta360_cam_front_optical_frame"


@requires_ffmpeg
@requires_real_capture
def test_metadata_topic_documents_the_imu_frame_and_lens_mapping(converted):
    from mcap_ros2.reader import read_ros2_messages

    _, path = converted
    payload = None
    for message in read_ros2_messages(path, topics=["/insta360/metadata"]):
        payload = json.loads(message.ros_msg.data)
    assert payload is not None
    assert payload["model"] == "Insta360 X5"
    assert "REP 103" in payload["imu"]["axes"]
    assert [lens["video_stream_index"] for lens in payload["lenses"]] == [0, 1]
    assert payload["lenses"][0]["topic"].endswith("cam_front/image/compressed")


@requires_ffmpeg
@requires_real_capture
def test_preview_image_is_a_full_rgb8_equirect_frame(converted):
    from mcap_ros2.reader import read_ros2_messages

    _, path = converted
    for message in read_ros2_messages(path, topics=["/insta360/preview/image"]):
        image = message.ros_msg
        assert image.encoding == "rgb8"
        assert image.width == 2 * image.height, "equirectangular frames are 2:1"
        assert image.step == image.width * 3
        assert len(image.data) == image.step * image.height
        assert image.is_bigendian == 0


@requires_ffmpeg
@requires_real_capture
def test_timestamps_land_on_the_capture_wall_clock(converted):
    summary, _ = converted
    # 2026-09-09T11:28:28Z, from the camera's own clock reference.
    assert 1_788_953_300 < summary.start_ns / 1e9 < 1_788_953_320


@requires_ffmpeg
@requires_real_capture
def test_relative_time_starts_the_bag_at_zero(tmp_path):
    output = tmp_path / "relative.mcap"
    summary = convert(
        Options(
            input_path=REAL_CAPTURE,
            output_path=str(output),
            max_frames=1,
            jpeg_quality=20,
            relative_time=True,
            include_audio=False,
            include_preview=False,
        )
    )
    assert summary.start_ns == 0


@requires_ffmpeg
@requires_real_capture
def test_streams_can_be_switched_off_individually(tmp_path):
    output = tmp_path / "imu-only.mcap"
    summary = convert(
        Options(
            input_path=REAL_CAPTURE,
            output_path=str(output),
            include_video=False,
            include_camera_info=False,
            include_exposure=False,
            include_preview=False,
            include_audio=False,
            include_tf=False,
        )
    )
    assert set(summary.message_counts) == {"/insta360/imu", "/insta360/metadata"}


@requires_ffmpeg
@requires_real_capture
def test_scaling_rescales_both_the_frames_and_the_intrinsics(tmp_path):
    from mcap_ros2.reader import read_ros2_messages

    output = tmp_path / "scaled.mcap"
    convert(
        Options(
            input_path=REAL_CAPTURE,
            output_path=str(output),
            max_frames=1,
            jpeg_quality=20,
            scale="0.25",
            include_audio=False,
            include_preview=False,
            include_imu=False,
        )
    )
    for message in read_ros2_messages(
        str(output), topics=["/insta360/cam_front/camera_info"]
    ):
        info = message.ros_msg
        assert (info.width, info.height) == (720, 720)
        # cx must track the smaller frame, not stay at the full-size value.
        assert info.k[2] == pytest.approx(360, abs=5)
    for message in read_ros2_messages(
        str(output), topics=["/insta360/cam_front/image/compressed"]
    ):
        data = bytes(message.ros_msg.data)
        index = data.index(b"\xff\xc0")
        height, width = struct.unpack(">HH", data[index + 5 : index + 9])
        assert (width, height) == (720, 720)


@requires_ffmpeg
@requires_real_capture
def test_swapping_lenses_exchanges_the_two_image_topics(tmp_path):
    from mcap_ros2.reader import read_ros2_messages

    def first_frame(swap):
        output = tmp_path / f"swap-{swap}.mcap"
        convert(
            Options(
                input_path=REAL_CAPTURE,
                output_path=str(output),
                max_frames=1,
                jpeg_quality=20,
                swap_lenses=swap,
                include_audio=False,
                include_preview=False,
                include_imu=False,
                include_exposure=False,
            )
        )
        for message in read_ros2_messages(
            str(output), topics=["/insta360/cam_front/image/compressed"]
        ):
            return bytes(message.ros_msg.data)
        raise AssertionError("no frame written")

    assert first_frame(False) != first_frame(True)


@requires_ffmpeg
@requires_real_capture
def test_tf_static_links_the_lidar_frame_to_both_image_frames(tmp_path):
    from mcap_ros2.reader import read_ros2_messages

    output = tmp_path / "tf.mcap"
    convert(
        Options(
            input_path=REAL_CAPTURE,
            output_path=str(output),
            max_frames=1,
            jpeg_quality=20,
            include_audio=False,
            include_preview=False,
            include_imu=False,
            lidar_frame="os_sensor",
            camera_xyz=(0.05, 0.0, 0.2),
        )
    )
    image_frames = {
        message.channel.topic: message.ros_msg.header.frame_id
        for message in read_ros2_messages(str(output))
        if message.channel.topic.endswith("/image/compressed")
    }
    (message,) = list(read_ros2_messages(str(output), topics=["/tf_static"]))
    transforms = message.ros_msg.transforms
    assert {t.child_frame_id for t in transforms} == set(image_frames.values())
    for transform in transforms:
        assert transform.header.frame_id == "os_sensor"
        translation = transform.transform.translation
        assert (translation.x, translation.y, translation.z) == (0.05, 0.0, 0.2)


@requires_ffmpeg
@requires_real_capture
def test_topic_prefix_is_applied(tmp_path):
    output = tmp_path / "prefixed.mcap"
    summary = convert(
        Options(
            input_path=REAL_CAPTURE,
            output_path=str(output),
            topic_prefix="/x5",
            include_video=False,
            include_camera_info=False,
            include_exposure=False,
            include_preview=False,
            include_audio=False,
        )
    )
    # /tf_static is fixed by tf itself, so it is the one topic the prefix does not touch.
    assert "/tf_static" in summary.message_counts
    assert all(
        topic.startswith("/x5/")
        for topic in summary.message_counts
        if topic != "/tf_static"
    )


@requires_ffmpeg
@requires_real_capture
@pytest.mark.parametrize("compression", ["none", "lz4", "zstd"])
def test_each_compression_setting_produces_a_readable_bag(tmp_path, compression):
    from mcap_ros2.reader import read_ros2_messages

    output = tmp_path / f"{compression}.mcap"
    convert(
        Options(
            input_path=REAL_CAPTURE,
            output_path=str(output),
            compression=compression,
            include_video=False,
            include_camera_info=False,
            include_exposure=False,
            include_preview=False,
            include_audio=False,
        )
    )
    count = sum(1 for _ in read_ros2_messages(str(output), topics=["/insta360/imu"]))
    assert count == 16752


# -- external calibration ---------------------------------------------------


def _camera(name, calibration=None, pose=None, path=None):
    return _Camera(
        name=name,
        video_index=0 if name == "cam_front" else 1,
        image_topic=f"/insta360/{name}/image/compressed",
        info_topic=f"/insta360/{name}/camera_info",
        frame_id=f"insta360_{name}_optical_frame",
        width=2880,
        height=2880,
        calibration=calibration,
        pose=pose,
        calibration_path=path,
    )


def test_a_calibration_for_a_lens_that_does_not_exist_is_rejected():
    options = Options(
        input_path="in.insv",
        output_path="out.mcap",
        hdmapping_calibration={"cam_left": "cam_left.json"},
    )
    with pytest.raises(ValueError, match="no such lens: cam_left"):
        _external_calibration(options, metadata_mod.Metadata(), Summary("out.mcap"))


def test_a_calibration_for_another_camera_is_applied_but_flagged(tmp_path):
    path = tmp_path / "cam_front.json"
    path.write_text(json.dumps(X5_CAM_FRONT))
    summary = Summary("out.mcap")
    options = Options(
        input_path="in.insv",
        output_path="out.mcap",
        hdmapping_calibration={"cam_front": str(path)},
    )
    loaded = _external_calibration(
        options, metadata_mod.Metadata(serial="SOMEOTHERBODY"), summary
    )
    assert set(loaded) == {"cam_front"}
    assert any("IAHEA2503XUXRF" in warning for warning in summary.warnings)


def test_a_matching_serial_says_nothing(tmp_path):
    path = tmp_path / "cam_front.json"
    path.write_text(json.dumps(X5_CAM_FRONT))
    summary = Summary("out.mcap")
    _external_calibration(
        Options(
            input_path="in.insv",
            output_path="out.mcap",
            hdmapping_calibration={"cam_front": str(path)},
        ),
        metadata_mod.Metadata(serial="IAHEA2503XUXRF"),
        summary,
    )
    assert summary.warnings == []


def test_swapping_lenses_alongside_a_calibration_is_flagged(tmp_path):
    path = tmp_path / "cam_front.json"
    path.write_text(json.dumps(X5_CAM_FRONT))
    summary = Summary("out.mcap")
    _external_calibration(
        Options(
            input_path="in.insv",
            output_path="out.mcap",
            swap_lenses=True,
            hdmapping_calibration={"cam_front": str(path)},
        ),
        metadata_mod.Metadata(serial="IAHEA2503XUXRF"),
        summary,
    )
    assert any("--swap-lenses" in warning for warning in summary.warnings)


def test_an_unreadable_calibration_stops_the_conversion(tmp_path):
    path = tmp_path / "cam_front.json"
    path.write_text("{}")
    with pytest.raises(ValueError):
        _external_calibration(
            Options(
                input_path="in.insv",
                output_path="out.mcap",
                hdmapping_calibration={"cam_front": str(path)},
            ),
            metadata_mod.Metadata(),
            Summary("out.mcap"),
        )


# -- what the bag says about where its calibration came from ----------------


def test_metadata_reports_the_source_of_each_lens_pose(tmp_path):
    path = tmp_path / "cam_front.json"
    path.write_text(json.dumps(X5_CAM_FRONT))
    calibration = hdmapping.load(str(path))
    cameras = [
        _camera("cam_front", calibration.lens(0, 2880, 2880), calibration.pose, str(path)),
        _camera("cam_back"),
    ]
    options = Options(input_path="in.insv", output_path="out.mcap")
    transforms = static_transforms(
        [(c.name, c.frame_id) for c in cameras], 0, "lidar",
        options.camera_xyz, options.camera_rpy, poses={"cam_front": calibration.pose},
    )
    payload = _metadata_payload(
        metadata_mod.Metadata(serial="IAHEA2503XUXRF"), cameras, options, transforms
    )

    front, back = payload["lenses"]
    assert front["extrinsics"]["calibration_source"] == f"hdmapping {path}"
    assert front["extrinsics"]["parent_frame"] == "lidar"
    assert front["extrinsics"]["published_on"] == TF_STATIC_TOPIC
    assert front["extrinsics"]["transform"]["rotation"]["w"] == pytest.approx(
        calibration.pose.rotation[3]
    )
    assert front["intrinsics"]["calibration_source"] == f"hdmapping {path}"
    # "source" there already names the lens model, and a dict literal silently keeps
    # only the last of two equal keys.
    assert front["intrinsics"]["source"]["model"].startswith("unified")
    # The camera's own stitching pose has no counterpart in an external calibration.
    assert front["intrinsics"]["rotation_deg"] is None

    assert back["extrinsics"]["calibration_source"] == "nominal, from --camera-xyz and --camera-rpy"
    assert back["intrinsics"] is None
    # The whole payload has to survive the json round trip it is written with.
    assert json.loads(json.dumps(payload, sort_keys=True))["lenses"][0] == front


def test_metadata_reports_extrinsics_that_are_not_published(tmp_path):
    cameras = [_camera("cam_front")]
    options = Options(input_path="in.insv", output_path="out.mcap", include_tf=False)
    transforms = static_transforms(
        [(c.name, c.frame_id) for c in cameras], 0, "lidar",
        options.camera_xyz, options.camera_rpy,
    )
    payload = _metadata_payload(metadata_mod.Metadata(), cameras, options, transforms)
    assert payload["lenses"][0]["extrinsics"]["published_on"] is None


def test_the_camera_serial_is_written_as_an_mcap_metadata_record(tmp_path):
    path = tmp_path / "meta.mcap"
    with open(path, "wb") as handle:
        writer = McapRos2Writer(handle)
        _write_camera_metadata(
            writer,
            metadata_mod.Metadata(
                serial="IAHEA2503XUXRF",
                model="Insta360 X5",
                firmware="v1.11.10_build1",
                capture_datetime="2026-09-09T13:28:27",
            ),
            [_camera("cam_front", path="/tmp/cam_front.json"), _camera("cam_back")],
        )
        writer.finish()

    with open(path, "rb") as handle:
        records = list(make_reader(handle).iter_metadata())
    (record,) = records
    assert record.name == CAMERA_METADATA_RECORD
    assert record.metadata["serial"] == "IAHEA2503XUXRF"
    assert record.metadata["model"] == "Insta360 X5"
    assert record.metadata["firmware"] == "v1.11.10_build1"
    assert record.metadata["captured"] == "2026-09-09T13:28:27"
    # Which lenses carry an external calibration is part of the bag's provenance.
    assert record.metadata["cam_front_calibration"] == "/tmp/cam_front.json"
    assert "cam_back_calibration" not in record.metadata


def test_the_metadata_record_tolerates_a_capture_that_names_nothing(tmp_path):
    path = tmp_path / "meta.mcap"
    with open(path, "wb") as handle:
        writer = McapRos2Writer(handle)
        _write_camera_metadata(writer, metadata_mod.Metadata(), [])
        writer.finish()
    with open(path, "rb") as handle:
        (record,) = list(make_reader(handle).iter_metadata())
    assert record.metadata == {"serial": "", "model": "", "firmware": "", "captured": ""}
