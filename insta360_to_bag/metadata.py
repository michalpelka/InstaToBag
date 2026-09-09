"""Typed view over the Insta360 protobuf metadata record (trailer record 0x0101).

Field numbers were recovered by decoding the wire format of real captures; the camera
ships no schema, so only the fields below are interpreted and everything else is left
alone.  Anything absent comes back as ``None`` and the caller degrades gracefully.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .calibration import LensCalibration, parse_offset_v2
from .protobuf import Message

# -- metadata field numbers -------------------------------------------------
F_SERIAL = 1
F_MODEL = 2
F_FIRMWARE = 3
F_OFFSET = 5
F_CAPTURE_DATETIME = 7  # YYYYMMDDhhmmss as a single integer, in camera local time
F_TRAILER_BASE = 9
F_FRAME_SIZE = 19  # sub-message {1: width, 2: height}
F_FPS = 20
F_PROFILE = 22
F_FIRST_FRAME_US = 24
F_SOURCE = 26  # sub-message, {3: original path on the SD card}
F_NATIVE_SIZE = 27
F_GAIN = 28
F_START_EPOCH_MS = 36
F_OFFSET_V2 = 54
F_SENSOR_RANGES = 65  # sub-message {1: accel full scale in g, 2: gyro full scale in dps}
F_DURATION_MS = 93
F_CLOCK = 98
F_OFFSET_V3 = 111

# -- F_CLOCK sub-message ----------------------------------------------------
C_FIRST_FRAME_US = 1
C_LAST_FRAME_US = 2
C_START_EPOCH_MS = 3
C_END_EPOCH_MS = 4
C_FRAME_COUNT = 5

DEFAULT_ACCEL_RANGE_G = 32.0
DEFAULT_GYRO_RANGE_DPS = 2000.0


@dataclass(frozen=True)
class Clock:
    """Maps the camera's monotonic microsecond clock onto wall-clock epoch time.

    Every timestamp in the trailer (IMU samples, exposure entries) is in microseconds
    since the camera booted.  The metadata pins one point of that clock -- the first
    encoded video frame -- to a UTC epoch value, which is what lets the bag carry real
    timestamps rather than an arbitrary origin.
    """

    first_frame_us: int
    first_frame_epoch_ns: int

    def to_epoch_ns(self, device_us: int) -> int:
        return self.first_frame_epoch_ns + (device_us - self.first_frame_us) * 1000


@dataclass
class Metadata:
    serial: Optional[str] = None
    model: Optional[str] = None
    firmware: Optional[str] = None
    profile: Optional[str] = None
    source_path: Optional[str] = None
    capture_datetime: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    frame_count: Optional[int] = None
    duration_s: Optional[float] = None
    first_frame_us: Optional[int] = None
    last_frame_us: Optional[int] = None
    start_epoch_ms: Optional[int] = None
    end_epoch_ms: Optional[int] = None
    accel_range_g: float = DEFAULT_ACCEL_RANGE_G
    gyro_range_dps: float = DEFAULT_GYRO_RANGE_DPS
    sensor_ranges_known: bool = False
    calibration_strings: Dict[str, str] = field(default_factory=dict)
    lenses: List[LensCalibration] = field(default_factory=list)

    # -- derived -----------------------------------------------------------

    @property
    def clock(self) -> Optional[Clock]:
        if self.first_frame_us is None or self.start_epoch_ms is None:
            return None
        return Clock(
            first_frame_us=self.first_frame_us,
            first_frame_epoch_ns=self.start_epoch_ms * 1_000_000,
        )

    @property
    def frame_interval_us(self) -> Optional[float]:
        if self.fps:
            return 1_000_000.0 / self.fps
        return None

    def as_dict(self) -> Dict[str, object]:
        """A JSON-serialisable summary, republished on the metadata topic."""
        return {
            "serial": self.serial,
            "model": self.model,
            "firmware": self.firmware,
            "profile": self.profile,
            "source_path": self.source_path,
            "capture_datetime": self.capture_datetime,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "duration_s": self.duration_s,
            "first_frame_us": self.first_frame_us,
            "last_frame_us": self.last_frame_us,
            "start_epoch_ms": self.start_epoch_ms,
            "end_epoch_ms": self.end_epoch_ms,
            "accel_range_g": self.accel_range_g,
            "gyro_range_dps": self.gyro_range_dps,
            "sensor_ranges_known": self.sensor_ranges_known,
            "calibration": self.calibration_strings,
        }


def _format_capture_datetime(value: Optional[int]) -> Optional[str]:
    """Render the packed YYYYMMDDhhmmss integer as an ISO-8601-ish local timestamp."""
    if value is None:
        return None
    text = str(value)
    if len(text) != 14:
        return text
    return f"{text[0:4]}-{text[4:6]}-{text[6:8]}T{text[8:10]}:{text[10:12]}:{text[12:14]}"


def parse(raw: bytes) -> Metadata:
    """Decode trailer record 0x0101 into a :class:`Metadata`."""
    message = Message.parse(raw)
    meta = Metadata(
        serial=message.text(F_SERIAL),
        model=message.text(F_MODEL),
        firmware=message.text(F_FIRMWARE),
        profile=message.text(F_PROFILE),
        capture_datetime=_format_capture_datetime(message.uint(F_CAPTURE_DATETIME)),
        fps=float(message.uint(F_FPS)) if message.uint(F_FPS) else None,
        first_frame_us=message.uint(F_FIRST_FRAME_US),
        start_epoch_ms=message.uint(F_START_EPOCH_MS),
    )

    source = message.message(F_SOURCE)
    if source is not None:
        meta.source_path = source.text(3)

    size = message.message(F_FRAME_SIZE)
    if size is not None:
        meta.width = size.uint(1)
        meta.height = size.uint(2)

    duration_ms = message.uint(F_DURATION_MS)
    if duration_ms is not None:
        meta.duration_s = duration_ms / 1000.0

    # The clock sub-message is the most precise source for all four of these, so it
    # overrides the standalone fields above where present.
    clock = message.message(F_CLOCK)
    if clock is not None:
        meta.first_frame_us = clock.uint(C_FIRST_FRAME_US, meta.first_frame_us)
        meta.last_frame_us = clock.uint(C_LAST_FRAME_US)
        meta.start_epoch_ms = clock.uint(C_START_EPOCH_MS, meta.start_epoch_ms)
        meta.end_epoch_ms = clock.uint(C_END_EPOCH_MS)
        meta.frame_count = clock.uint(C_FRAME_COUNT)

    ranges = message.message(F_SENSOR_RANGES)
    if ranges is not None:
        accel = ranges.uint(1)
        gyro = ranges.uint(2)
        if accel and gyro:
            meta.accel_range_g = float(accel)
            meta.gyro_range_dps = float(gyro)
            meta.sensor_ranges_known = True

    for name, number in (
        ("offset", F_OFFSET),
        ("offset_v2", F_OFFSET_V2),
        ("offset_v3", F_OFFSET_V3),
    ):
        text = message.text(number)
        if text:
            meta.calibration_strings[name] = text

    v2 = meta.calibration_strings.get("offset_v2")
    if v2 and meta.width and meta.height:
        meta.lenses = parse_offset_v2(v2, meta.width, meta.height)

    return meta
