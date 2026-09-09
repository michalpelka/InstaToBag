import math
import struct

import pytest

from insta360_to_bag import sensors


def imu_record(samples):
    """Pack (timestamp, 6 offset-binary channel counts) tuples as record 0x0003."""
    out = bytearray()
    for device_us, channels in samples:
        out += struct.pack("<Q6H", device_us, *channels)
    return bytes(out)


def test_imu_channels_are_offset_binary_and_scaled_to_ros_units():
    # One count above zero on every channel.
    raw = imu_record([(1000, (32769, 32768, 32768, 32769, 32768, 32768))])
    sample = sensors.read_imu(raw, accel_range_g=32.0, gyro_range_dps=2000.0)[0]
    assert sample.device_us == 1000
    assert sample.accel[0] == pytest.approx(32.0 * sensors.STANDARD_GRAVITY / 32768)
    assert sample.accel[1] == 0.0
    assert sample.gyro[0] == pytest.approx(math.radians(2000.0) / 32768)
    assert sample.gyro[1] == 0.0


def test_counts_below_the_zero_point_go_negative():
    raw = imu_record([(0, (32768 - 1024, 32768, 32768, 32768, 32768, 32768))])
    sample = sensors.read_imu(raw, 32.0, 2000.0)[0]
    # 1024 counts at +/-32 g full scale is exactly one g.
    assert sample.accel[0] == pytest.approx(-sensors.STANDARD_GRAVITY)


def test_full_scale_is_honoured():
    raw = imu_record([(0, (32768 + 1024, 32768, 32768, 32768, 32768, 32768))])
    at_32g = sensors.read_imu(raw, 32.0, 2000.0)[0]
    at_16g = sensors.read_imu(raw, 16.0, 2000.0)[0]
    assert at_32g.accel[0] == pytest.approx(2 * at_16g.accel[0])


def test_accelerometer_is_the_first_triple():
    # A one-g reading on accel z and nothing on the gyro.
    raw = imu_record([(0, (32768, 32768, 32768 + 1024, 32768, 32768, 32768))])
    sample = sensors.read_imu(raw, 32.0, 2000.0)[0]
    assert sample.accel[2] == pytest.approx(sensors.STANDARD_GRAVITY)
    assert sample.gyro == (0.0, 0.0, 0.0)


def test_trailing_partial_sample_is_ignored():
    raw = imu_record([(0, (32768,) * 6)]) + b"\x00" * 7
    assert len(sensors.read_imu(raw, 32.0, 2000.0)) == 1
    assert sensors.read_imu(b"", 32.0, 2000.0) == []


def test_exposure_record_decodes():
    raw = struct.pack("<Qd", 32271386, 0.004995) + struct.pack("<Qd", 32313054, 0.006)
    samples = sensors.read_exposure(raw)
    assert [s.device_us for s in samples] == [32271386, 32313054]
    assert samples[0].exposure_s == pytest.approx(0.004995)


# -- frame timing -----------------------------------------------------------


def exposure_samples(start_us, count, step_us, value=0.005):
    return [
        sensors.ExposureSample(start_us + index * step_us, value)
        for index in range(count)
    ]


def test_frame_timestamps_prefer_the_exposure_record():
    # Two pre-roll entries before the first encoded frame, as real files carry.
    exposures = exposure_samples(30000, 6, 41661)
    first_frame_us = 30000 + 2 * 41661
    stamps, values = sensors.frame_timestamps(
        exposures, first_frame_us, frame_count=3, frame_interval_us=41666.67
    )
    assert stamps == [first_frame_us, first_frame_us + 41661, first_frame_us + 2 * 41661]
    assert values == [0.005, 0.005, 0.005]


def test_frame_timestamps_fall_back_to_the_nominal_rate():
    stamps, values = sensors.frame_timestamps(
        [], first_frame_us=1000, frame_count=3, frame_interval_us=41666.67
    )
    assert stamps == [1000, 1000 + 41667, 1000 + 83333]
    assert values == [None, None, None]


def test_short_exposure_coverage_falls_back_rather_than_truncating():
    exposures = exposure_samples(1000, 2, 41661)
    stamps, values = sensors.frame_timestamps(
        exposures, first_frame_us=1000, frame_count=5, frame_interval_us=41666.67
    )
    assert len(stamps) == 5
    assert values == [None] * 5


def test_frame_timestamps_without_any_reference_raises():
    with pytest.raises(ValueError, match="cannot establish frame timestamps"):
        sensors.frame_timestamps([], None, frame_count=3, frame_interval_us=None)


# -- preview ----------------------------------------------------------------


def preview_record(width, height, payload=None):
    header = struct.pack("<10I", 1, 0, 1, 0, width, height, 0, 0, 0, 0)
    body = payload if payload is not None else b"\x80" * (width * height * 3 // 2)
    return header + body


def test_preview_header_and_payload():
    preview = sensors.read_preview(preview_record(64, 32))
    assert (preview.width, preview.height) == (64, 32)
    assert len(preview.nv12) == 64 * 32 * 3 // 2


def test_preview_trims_trailing_padding():
    raw = preview_record(8, 8) + b"\xff" * 100
    preview = sensors.read_preview(raw)
    assert len(preview.nv12) == 8 * 8 * 3 // 2
    assert set(preview.nv12) == {0x80}


@pytest.mark.parametrize(
    "raw, match",
    [
        (b"\x00" * 8, "too short"),
        (preview_record(64, 32, payload=b"\x00" * 10), "carries"),
        (preview_record(0, 0), "carries"),
    ],
)
def test_bad_preview_records_raise(raw, match):
    with pytest.raises(ValueError, match=match):
        sensors.read_preview(raw)
