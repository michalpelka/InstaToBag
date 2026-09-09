import shutil
import struct

import pytest

from insta360_to_bag import media

from conftest import REAL_CAPTURE, requires_real_capture

requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not on PATH",
)


def segment(marker, payload):
    return bytes([0xFF, marker]) + struct.pack(">H", len(payload) + 2) + payload


def jpeg(scan=b"\x12\x34", comment=b"c"):
    return (
        b"\xff\xd8"
        + segment(0xE0, b"JFIF\x00")
        + segment(0xFE, comment)  # COM
        + segment(0xDA, b"\x01\x00")  # SOS header
        + scan
        + b"\xff\xd9"
    )


def test_finds_the_end_of_a_simple_jpeg():
    data = jpeg()
    assert media._find_jpeg_end(bytearray(data), 0) == len(data)


def test_byte_stuffed_ff00_in_scan_data_is_not_a_marker():
    data = jpeg(scan=b"\x12\xff\x00\x34\xff\x00")
    assert media._find_jpeg_end(bytearray(data), 0) == len(data)


def test_restart_markers_in_scan_data_are_skipped():
    data = jpeg(scan=b"\x12\xff\xd0\x34\xff\xd7\x56")
    assert media._find_jpeg_end(bytearray(data), 0) == len(data)


def test_ffd9_inside_a_segment_payload_does_not_end_the_image():
    # A legal COM segment may contain any bytes, including FF D9.  Searching for a
    # bare FF D9 would cut the frame short here; walking the markers does not.
    data = jpeg(comment=b"before\xff\xd9after")
    assert media._find_jpeg_end(bytearray(data), 0) == len(data)
    assert bytearray(data).find(b"\xff\xd9") < len(data) - 2


def test_fill_bytes_before_a_marker_are_tolerated():
    data = (
        b"\xff\xd8"
        + segment(0xDA, b"\x01\x00")
        + b"\x12\x34"
        + b"\xff\xff\xff\xd9"  # 0xFF fill bytes ahead of EOI
    )
    assert media._find_jpeg_end(bytearray(data), 0) == len(data)


def test_two_concatenated_images_are_split_at_the_right_place():
    first, second = jpeg(scan=b"\xaa"), jpeg(scan=b"\xbb\xcc")
    buf = bytearray(first + second)
    end = media._find_jpeg_end(buf, 0)
    assert bytes(buf[:end]) == first
    del buf[:end]
    assert media._find_jpeg_end(buf, 0) == len(second)


@pytest.mark.parametrize("cut", [3, 8, 14])
def test_truncated_input_reports_incomplete(cut):
    data = jpeg(scan=b"\x12\x34\x56\x78")
    assert media._find_jpeg_end(bytearray(data[:cut]), 0) == -1


def test_not_starting_at_soi_raises():
    with pytest.raises(media.FFmpegError, match="SOI"):
        media._find_jpeg_end(bytearray(b"\x00\x01\x02\x03"), 0)


def test_zero_length_segment_is_rejected():
    data = b"\xff\xd8" + b"\xff\xe0\x00\x00" + b"\xff\xd9"
    with pytest.raises(media.FFmpegError, match="segment length"):
        media._find_jpeg_end(bytearray(data), 0)


def test_require_tools_passes_when_ffmpeg_is_installed(monkeypatch):
    monkeypatch.setattr(media.shutil, "which", lambda name: "/usr/bin/" + name)
    media.require_tools()


def test_require_tools_names_what_is_missing(monkeypatch):
    monkeypatch.setattr(media.shutil, "which", lambda name: None)
    with pytest.raises(media.FFmpegError, match="ffmpeg and ffprobe not found"):
        media.require_tools()


@pytest.mark.parametrize(
    "text, expected",
    [("24/1", 24.0), ("30000/1001", pytest.approx(29.97, abs=0.01)), ("0/0", None),
     ("", None), (None, None), ("bad/rate", None)],
)
def test_frame_rate_parsing(text, expected):
    assert media._parse_rate(text) == expected


@requires_ffmpeg
def test_nv12_to_rgb8_round_trips_a_flat_grey_frame():
    width, height = 16, 8
    # Y=128 with neutral chroma is a mid grey.
    nv12 = bytes([128] * (width * height)) + bytes([128] * (width * height // 2))
    rgb = media.nv12_to_rgb8(nv12, width, height)
    assert len(rgb) == width * height * 3
    assert all(abs(value - 128) < 12 for value in rgb)


@requires_ffmpeg
def test_nv12_to_rgb8_rejects_a_short_buffer():
    with pytest.raises(media.FFmpegError):
        media.nv12_to_rgb8(b"\x00" * 10, 64, 64)


@requires_ffmpeg
@requires_real_capture
def test_probe_reports_both_fisheye_tracks_and_the_audio_track():
    probe = media.probe(REAL_CAPTURE)
    assert len(probe.video) == 2
    assert all(stream.width == 2880 and stream.height == 2880 for stream in probe.video)
    assert all(stream.codec == "hevc" for stream in probe.video)
    assert len(probe.audio) == 1
    assert probe.audio[0].sample_rate == 48000 and probe.audio[0].channels == 2
    assert probe.duration_s == pytest.approx(15.58, abs=0.1)


@requires_ffmpeg
def test_probe_on_a_non_media_file_raises(tmp_path):
    path = tmp_path / "not-media.bin"
    path.write_bytes(b"\x00" * 1024)
    with pytest.raises(media.FFmpegError, match="ffprobe failed"):
        media.probe(str(path))


@requires_ffmpeg
@requires_real_capture
def test_frame_pipe_yields_complete_jpegs():
    with media.FramePipe(REAL_CAPTURE, video_index=0, quality=8, limit=3) as pipe:
        frames = list(pipe)
    assert len(frames) == 3
    for frame in frames:
        assert frame.startswith(b"\xff\xd8") and frame.endswith(b"\xff\xd9")
        assert len(frame) > 10_000


@requires_ffmpeg
@requires_real_capture
def test_frame_pipe_honours_a_scale_filter():
    with media.FramePipe(
        REAL_CAPTURE, video_index=0, quality=8, scale_filter="320:320", limit=1
    ) as pipe:
        frame = next(iter(pipe))
    # SOF0 carries the real decoded dimensions; confirm ffmpeg actually scaled.
    index = frame.index(b"\xff\xc0")
    height, width = struct.unpack(">HH", frame[index + 5 : index + 9])
    assert (width, height) == (320, 320)


@requires_ffmpeg
@requires_real_capture
def test_audio_pipe_blocks_are_whole_frames_and_indexed_in_order():
    with media.AudioPipe(
        REAL_CAPTURE, audio_index=0, sample_rate=48000, channels=2,
        samples_per_chunk=2000, duration_s=0.5,
    ) as pipe:
        blocks = list(pipe)
    assert blocks
    assert [index for index, _ in blocks] == [i * 2000 for i in range(len(blocks))]
    assert all(len(block) % 4 == 0 for _, block in blocks)
    assert sum(len(block) for _, block in blocks) == pytest.approx(0.5 * 48000 * 4, rel=0.05)
