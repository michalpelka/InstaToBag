import os
import shutil

import pytest

from insta360_to_bag.cli import (
    _human_bytes,
    build_parser,
    main,
    parse_calibration_args,
)

from conftest import REAL_CAPTURE, requires_real_capture

requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not on PATH",
)


def test_content_flags_default_to_on_and_switch_off():
    args = build_parser().parse_args(["in.insv"])
    assert (args.video, args.camera_info, args.imu, args.exposure, args.preview,
            args.audio, args.tf) == (True,) * 7
    args = build_parser().parse_args(["in.insv", "--no-imu", "--no-audio", "--no-tf"])
    assert args.imu is False and args.audio is False and args.tf is False
    assert args.video is True


def test_defaults_match_the_documented_values():
    args = build_parser().parse_args(["in.insv"])
    assert args.topic_prefix == "/insta360"
    assert args.compression == "zstd"
    assert args.jpeg_quality == 3
    assert args.scale is None and args.max_frames is None
    assert args.relative_time is False
    assert args.lidar_frame == "lidar"
    assert args.camera_xyz == [0.0, 0.0, 0.0]
    assert args.camera_rpy == [90.0, 0.0, 90.0]


def test_camera_position_accepts_negative_coordinates():
    args = build_parser().parse_args(["in.insv", "--camera-xyz", "-0.1", "0.2", "-0.35"])
    assert args.camera_xyz == [-0.1, 0.2, -0.35]


def test_camera_orientation_accepts_negative_angles():
    args = build_parser().parse_args(["in.insv", "--camera-rpy", "0", "-5.5", "-90"])
    assert args.camera_rpy == [0.0, -5.5, -90.0]


def test_camera_position_needs_three_values():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["in.insv", "--camera-xyz", "0.1", "0.2"])


@pytest.mark.parametrize("frame", ["", "/lidar"])
def test_bad_lidar_frame_is_reported_before_any_work(tmp_path, capsys, frame):
    source = tmp_path / "x.insv"
    source.write_bytes(b"\x00" * 128)
    with pytest.raises(SystemExit):
        main([str(source), "--lidar-frame", frame])
    assert "--lidar-frame" in capsys.readouterr().err


def test_unknown_compression_is_rejected_at_parse_time():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["in.insv", "--compression", "brotli"])


def test_missing_input_is_reported(capsys):
    with pytest.raises(SystemExit):
        main(["/nonexistent/file.insv"])
    assert "not found" in capsys.readouterr().err


def test_bad_scale_is_reported_before_any_work(tmp_path, capsys):
    source = tmp_path / "x.insv"
    source.write_bytes(b"\x00" * 128)
    with pytest.raises(SystemExit):
        main([str(source), "--scale", "3.0"])
    assert "--scale" in capsys.readouterr().err


def test_existing_output_is_not_clobbered_without_force(tmp_path, capsys):
    source = tmp_path / "x.insv"
    source.write_bytes(b"\x00" * 128)
    output = tmp_path / "out.mcap"
    output.write_text("precious")
    assert main([str(source), "-o", str(output)]) == 2
    assert "--force" in capsys.readouterr().err
    assert output.read_text() == "precious"


def test_a_file_without_a_trailer_exits_with_an_error(tmp_path, capsys):
    source = tmp_path / "plain.mp4"
    source.write_bytes(b"\x00" * 4096)
    assert main([str(source), "-o", str(tmp_path / "out.mcap")]) == 1
    assert "error:" in capsys.readouterr().err


def test_default_output_path_sits_beside_the_input():
    parser = build_parser()
    args = parser.parse_args(["/tmp/VID_1.insv"])
    assert args.output is None
    assert os.path.splitext("/tmp/VID_1.insv")[0] + ".mcap" == "/tmp/VID_1.mcap"


@pytest.mark.parametrize(
    "size, expected",
    [(512, "512 B"), (2048, "2.0 KiB"), (5 << 20, "5.0 MiB"), (3 << 30, "3.0 GiB")],
)
def test_human_bytes(size, expected):
    assert _human_bytes(size) == expected


@requires_ffmpeg
@requires_real_capture
def test_inspect_reports_the_capture_without_writing_anything(tmp_path, capsys):
    before = set(os.listdir(tmp_path))
    assert main([REAL_CAPTURE, "--inspect"]) == 0
    out = capsys.readouterr().out
    assert "Insta360 X5" in out
    assert "imu (accel + gyro)" in out
    assert "1002.8 Hz" in out
    assert "calibration" in out
    assert set(os.listdir(tmp_path)) == before


@requires_ffmpeg
@requires_real_capture
def test_full_cli_run_writes_a_bag_and_reports_it(tmp_path, capsys):
    output = tmp_path / "cli.mcap"
    code = main([
        REAL_CAPTURE, "-o", str(output), "--max-frames", "2",
        "--jpeg-quality", "20", "--no-audio", "--no-preview",
    ])
    assert code == 0
    assert output.exists() and output.stat().st_size > 0
    out = capsys.readouterr().out
    assert "wrote" in out
    assert "/insta360/cam_front/image/compressed" in out


@requires_ffmpeg
@requires_real_capture
def test_force_overwrites_and_quiet_suppresses_progress(tmp_path, capsys):
    output = tmp_path / "cli.mcap"
    output.write_text("stale")
    code = main([
        REAL_CAPTURE, "-o", str(output), "--force", "--quiet",
        "--no-video", "--no-audio", "--no-preview", "--no-exposure",
    ])
    assert code == 0
    assert output.read_bytes()[:4] != b"stal"
    assert capsys.readouterr().out == ""


# -- external calibration arguments -----------------------------------------


def test_calibration_defaults_to_none_given():
    assert build_parser().parse_args(["in.insv"]).hdmapping_calibration == []
    assert parse_calibration_args([]) == {}


def test_a_file_named_after_its_lens_needs_no_prefix(tmp_path):
    front = tmp_path / "cam_front.json"
    back = tmp_path / "cam_back.json"
    front.write_text("{}")
    back.write_text("{}")
    assert parse_calibration_args([str(front), str(back)]) == {
        "cam_front": str(front),
        "cam_back": str(back),
    }


def test_an_explicit_lens_prefix_overrides_the_file_name(tmp_path):
    path = tmp_path / "2026-09-09-run3.json"
    path.write_text("{}")
    assert parse_calibration_args([f"cam_back={path}"]) == {"cam_back": str(path)}


def test_a_path_holding_an_equals_sign_is_still_a_path(tmp_path):
    directory = tmp_path / "run=3"
    directory.mkdir()
    path = directory / "cam_front.json"
    path.write_text("{}")
    assert parse_calibration_args([str(path)]) == {"cam_front": str(path)}


def test_a_file_whose_lens_cannot_be_told_is_an_error(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="could not tell which lens"):
        parse_calibration_args([str(path)])


def test_an_unknown_lens_name_is_an_error(tmp_path):
    path = tmp_path / "x.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="could not tell which lens"):
        parse_calibration_args([f"cam_left={path}"])


def test_two_files_for_one_lens_is_an_error(tmp_path):
    first = tmp_path / "cam_front.json"
    second = tmp_path / "other.json"
    first.write_text("{}")
    second.write_text("{}")
    with pytest.raises(ValueError, match="two calibration files"):
        parse_calibration_args([str(first), f"cam_front={second}"])


def test_a_missing_calibration_file_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        parse_calibration_args([str(tmp_path / "cam_front.json")])


def test_a_bad_calibration_argument_is_reported_before_any_work(tmp_path, capsys):
    source = tmp_path / "x.insv"
    source.write_bytes(b"\x00" * 128)
    with pytest.raises(SystemExit):
        main([str(source), "--hdmapping-calibration", str(tmp_path / "cam_front.json")])
    assert "not found" in capsys.readouterr().err
