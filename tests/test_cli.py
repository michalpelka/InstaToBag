import os
import shutil

import pytest

from insta360_to_bag.cli import _human_bytes, build_parser, main

from conftest import REAL_CAPTURE, requires_real_capture

requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not on PATH",
)


def test_content_flags_default_to_on_and_switch_off():
    args = build_parser().parse_args(["in.insv"])
    assert (args.video, args.camera_info, args.imu, args.exposure, args.preview,
            args.audio) == (True,) * 6
    args = build_parser().parse_args(["in.insv", "--no-imu", "--no-audio"])
    assert args.imu is False and args.audio is False
    assert args.video is True


def test_defaults_match_the_documented_values():
    args = build_parser().parse_args(["in.insv"])
    assert args.topic_prefix == "/insta360"
    assert args.compression == "zstd"
    assert args.jpeg_quality == 3
    assert args.scale is None and args.max_frames is None
    assert args.relative_time is False


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
