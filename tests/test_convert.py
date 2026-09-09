import json
import math
import shutil
import struct

import pytest

from insta360_to_bag.convert import Options, convert, resolve_scale

from conftest import REAL_CAPTURE, requires_real_capture

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
def test_camera_info_matches_the_images_and_carries_a_non_standard_model(converted):
    from insta360_to_bag.calibration import DISTORTION_MODEL
    from mcap_ros2.reader import read_ros2_messages

    _, path = converted
    for message in read_ros2_messages(path, topics=["/insta360/cam_front/camera_info"]):
        info = message.ros_msg
        assert (info.width, info.height) == (2880, 2880)
        assert info.distortion_model == DISTORTION_MODEL
        assert len(info.d) == 5 and len(info.k) == 9 and len(info.p) == 12
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
    assert all(topic.startswith("/x5/") for topic in summary.message_counts)


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
