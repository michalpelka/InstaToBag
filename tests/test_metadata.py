import pytest

from insta360_to_bag import metadata as metadata_mod

from conftest import REAL_CAPTURE, pb_bytes, pb_string, pb_varint, requires_real_capture
from test_calibration import X5_OFFSET_V2


def build_metadata(**overrides):
    clock = (
        pb_varint(metadata_mod.C_FIRST_FRAME_US, 32271386)
        + pb_varint(metadata_mod.C_LAST_FRAME_US, 47854719)
        + pb_varint(metadata_mod.C_START_EPOCH_MS, 1788953308040)
        + pb_varint(metadata_mod.C_END_EPOCH_MS, 1788953323623)
        + pb_varint(metadata_mod.C_FRAME_COUNT, 374)
    )
    raw = (
        pb_string(metadata_mod.F_SERIAL, "IAHEA2503XUXRF")
        + pb_string(metadata_mod.F_MODEL, "Insta360 X5")
        + pb_string(metadata_mod.F_FIRMWARE, "v1.10.11_build1")
        + pb_varint(metadata_mod.F_CAPTURE_DATETIME, 20260909132827)
        + pb_bytes(metadata_mod.F_FRAME_SIZE, pb_varint(1, 2880) + pb_varint(2, 2880))
        + pb_varint(metadata_mod.F_FPS, 24)
        + pb_string(metadata_mod.F_PROFILE, "standard")
        + pb_bytes(metadata_mod.F_SOURCE, pb_string(3, "/DCIM/Camera01/VID.insv"))
        + pb_string(metadata_mod.F_OFFSET_V2, X5_OFFSET_V2)
        + pb_bytes(metadata_mod.F_SENSOR_RANGES, pb_varint(1, 32) + pb_varint(2, 2000))
        + pb_varint(metadata_mod.F_DURATION_MS, 15583)
        + pb_bytes(metadata_mod.F_CLOCK, clock)
    )
    for extra in overrides.get("extra", ()):
        raw += extra
    if overrides.get("drop_clock"):
        raw = raw.replace(pb_bytes(metadata_mod.F_CLOCK, clock), b"")
    if overrides.get("drop_ranges"):
        raw = raw.replace(
            pb_bytes(metadata_mod.F_SENSOR_RANGES, pb_varint(1, 32) + pb_varint(2, 2000)),
            b"",
        )
    return raw


def test_identity_and_geometry():
    meta = metadata_mod.parse(build_metadata())
    assert meta.serial == "IAHEA2503XUXRF"
    assert meta.model == "Insta360 X5"
    assert meta.firmware == "v1.10.11_build1"
    assert meta.profile == "standard"
    assert meta.source_path == "/DCIM/Camera01/VID.insv"
    assert (meta.width, meta.height) == (2880, 2880)
    assert meta.fps == 24.0
    assert meta.frame_count == 374
    assert meta.duration_s == pytest.approx(15.583)


def test_capture_datetime_is_unpacked():
    meta = metadata_mod.parse(build_metadata())
    assert meta.capture_datetime == "2026-09-09T13:28:27"


def test_clock_maps_device_time_to_epoch_nanoseconds():
    meta = metadata_mod.parse(build_metadata())
    clock = meta.clock
    assert clock is not None
    # The pinned point maps exactly.
    assert clock.to_epoch_ns(32271386) == 1788953308040 * 1_000_000
    # One second of device time is one second of wall time.
    assert clock.to_epoch_ns(32271386 + 1_000_000) - clock.to_epoch_ns(32271386) == 10**9
    # And it extrapolates backwards, for IMU samples that precede the first frame.
    assert clock.to_epoch_ns(32271386 - 1_000_000) < 1788953308040 * 1_000_000


def test_frame_interval_follows_the_frame_rate():
    meta = metadata_mod.parse(build_metadata())
    assert meta.frame_interval_us == pytest.approx(1_000_000 / 24)


def test_sensor_ranges_are_flagged_when_present_and_when_assumed():
    known = metadata_mod.parse(build_metadata())
    assert (known.accel_range_g, known.gyro_range_dps) == (32.0, 2000.0)
    assert known.sensor_ranges_known is True

    assumed = metadata_mod.parse(build_metadata(drop_ranges=True))
    assert assumed.sensor_ranges_known is False
    assert assumed.accel_range_g == metadata_mod.DEFAULT_ACCEL_RANGE_G
    assert assumed.gyro_range_dps == metadata_mod.DEFAULT_GYRO_RANGE_DPS


def test_missing_clock_leaves_no_wall_clock_reference():
    meta = metadata_mod.parse(build_metadata(drop_clock=True))
    assert meta.clock is None
    assert meta.frame_count is None


def test_standalone_fields_are_used_when_the_clock_submessage_is_absent():
    extra = (
        pb_varint(metadata_mod.F_FIRST_FRAME_US, 32271386)
        + pb_varint(metadata_mod.F_START_EPOCH_MS, 1788953307924)
    )
    meta = metadata_mod.parse(build_metadata(drop_clock=True, extra=(extra,)))
    assert meta.first_frame_us == 32271386
    assert meta.clock is not None


def test_calibration_is_parsed_and_rescaled():
    meta = metadata_mod.parse(build_metadata())
    assert "offset_v2" in meta.calibration_strings
    assert len(meta.lenses) == 2
    assert meta.lenses[0].cx == pytest.approx(1440, abs=15)


def test_empty_metadata_parses_to_all_unknown():
    meta = metadata_mod.parse(b"")
    assert meta.serial is None and meta.clock is None and meta.lenses == []
    assert meta.as_dict()["model"] is None


def test_as_dict_is_json_serialisable():
    import json

    payload = metadata_mod.parse(build_metadata()).as_dict()
    assert json.loads(json.dumps(payload))["serial"] == "IAHEA2503XUXRF"


@requires_real_capture
def test_real_capture_metadata():
    from insta360_to_bag import trailer as trailer_mod

    with trailer_mod.Trailer(REAL_CAPTURE) as trailer:
        meta = metadata_mod.parse(trailer.read(trailer_mod.REC_METADATA))
    assert meta.model == "Insta360 X5"
    assert (meta.width, meta.height) == (2880, 2880)
    assert meta.frame_count == 374
    assert meta.sensor_ranges_known
    assert len(meta.lenses) == 2
    # The clock's two ends must agree on the clip length to within a frame.
    device_span = (meta.last_frame_us - meta.first_frame_us) / 1e6
    wall_span = (meta.end_epoch_ms - meta.start_epoch_ms) / 1e3
    assert device_span == pytest.approx(wall_span, abs=1 / 24)
