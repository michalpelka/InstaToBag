"""Decoders for the fixed-layout sensor records in the Insta360 trailer.

Layouts here were recovered by inspection and then validated physically rather than
just structurally -- see the notes on :func:`read_imu`.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import List, Optional, Tuple

#: Standard gravity, used to convert the accelerometer from g to m/s^2 (REP 145 wants
#: sensor_msgs/Imu in m/s^2 and rad/s).
STANDARD_GRAVITY = 9.80665

IMU_STRIDE = 20  # uint64 timestamp + 6 x uint16 channels
EXPOSURE_STRIDE = 16  # uint64 timestamp + float64 seconds
PREVIEW_HEADER_LEN = 40

#: The six IMU channels are offset-binary, i.e. a reading of 0 sits at 0x8000 rather
#: than being two's complement.  Interpreting them as int16 makes four of the six
#: channels wrap through +/-32768, which is how this was caught.
IMU_ZERO = 32768


@dataclass(frozen=True)
class ImuSample:
    """One IMU sample in ROS units: m/s^2 and rad/s, in the raw sensor frame."""

    device_us: int
    accel: Tuple[float, float, float]
    gyro: Tuple[float, float, float]


@dataclass(frozen=True)
class ExposureSample:
    device_us: int
    exposure_s: float


@dataclass(frozen=True)
class PreviewImage:
    width: int
    height: int
    nv12: bytes


def read_imu(
    raw: bytes,
    accel_range_g: float,
    gyro_range_dps: float,
) -> List[ImuSample]:
    """Decode trailer record 0x0003 into calibrated IMU samples.

    Layout is a packed array of ``<Q6H``: a microsecond device timestamp followed by
    accelerometer x/y/z then gyroscope x/y/z, each an offset-binary 16-bit count over
    the full-scale range reported in the metadata (typically +/-32 g and +/-2000 dps).

    That assignment is not guesswork: with these scales the accelerometer magnitude
    holds at 1.008 g across a whole handheld clip, and integrating the gyroscope over
    the same clip yields 1863 deg of total rotation against 1828 deg of gravity-vector
    travel measured independently from the accelerometer.  Swapping the two triples,
    or reading the counts as two's complement, breaks both checks.

    The axes are the sensor's own and are *not* rotated into REP 103 (x forward,
    y left, z up); the mapping from the X-series sensor frame to the camera body frame
    is undocumented, so the raw frame is preserved and named in the message header
    instead of being silently reinterpreted.
    """
    accel_scale = (accel_range_g * STANDARD_GRAVITY) / IMU_ZERO
    gyro_scale = math.radians(gyro_range_dps) / IMU_ZERO

    count = len(raw) // IMU_STRIDE
    samples: List[ImuSample] = []
    unpack = struct.Struct("<Q6H").unpack_from
    for index in range(count):
        device_us, ax, ay, az, gx, gy, gz = unpack(raw, index * IMU_STRIDE)
        samples.append(
            ImuSample(
                device_us=device_us,
                accel=(
                    (ax - IMU_ZERO) * accel_scale,
                    (ay - IMU_ZERO) * accel_scale,
                    (az - IMU_ZERO) * accel_scale,
                ),
                gyro=(
                    (gx - IMU_ZERO) * gyro_scale,
                    (gy - IMU_ZERO) * gyro_scale,
                    (gz - IMU_ZERO) * gyro_scale,
                ),
            )
        )
    return samples


def read_exposure(raw: bytes) -> List[ExposureSample]:
    """Decode trailer record 0x0004: ``<Qd`` of device timestamp and exposure seconds.

    The entries are emitted one per captured frame and start a little before the first
    *encoded* frame, so the caller aligns them by timestamp rather than by index.
    """
    count = len(raw) // EXPOSURE_STRIDE
    unpack = struct.Struct("<Qd").unpack_from
    return [
        ExposureSample(*unpack(raw, index * EXPOSURE_STRIDE)) for index in range(count)
    ]


def frame_timestamps(
    exposures: List[ExposureSample],
    first_frame_us: Optional[int],
    frame_count: int,
    frame_interval_us: Optional[float],
) -> Tuple[List[int], List[Optional[float]]]:
    """Work out a device timestamp for every encoded video frame.

    Prefers the exposure record, whose entries are stamped with the sensor's real
    (slightly non-nominal) frame interval and whose first in-range entry lands exactly
    on the metadata's first-frame timestamp.  Falls back to a uniform nominal interval
    when that record is missing or does not cover the clip.
    """
    if first_frame_us is not None and exposures:
        aligned = [s for s in exposures if s.device_us >= first_frame_us]
        if len(aligned) >= frame_count:
            chosen = aligned[:frame_count]
            return (
                [s.device_us for s in chosen],
                [s.exposure_s for s in chosen],
            )

    if first_frame_us is None or not frame_interval_us:
        raise ValueError(
            "cannot establish frame timestamps: no exposure record and no frame rate "
            "in the metadata"
        )
    stamps = [first_frame_us + round(index * frame_interval_us) for index in range(frame_count)]
    return stamps, [None] * frame_count


def read_preview(raw: bytes) -> PreviewImage:
    """Decode trailer record 0x0002: a 40-byte header then an NV12 equirect thumbnail.

    Header is a packed ``<10I``; words 4 and 5 are the width and height.  The pixel
    format is NV12 (a full-size luma plane followed by interleaved Cb/Cr at half
    resolution) -- confirmed by rendering, since NV21 comes out with the chroma
    channels swapped and I420 comes out desaturated.
    """
    if len(raw) < PREVIEW_HEADER_LEN:
        raise ValueError("preview record is too short to hold a header")
    header = struct.unpack_from("<10I", raw)
    width, height = header[4], header[5]
    expected = width * height * 3 // 2
    payload = raw[PREVIEW_HEADER_LEN:]
    if width <= 0 or height <= 0 or len(payload) < expected:
        raise ValueError(
            f"preview record claims {width}x{height} NV12 ({expected} bytes) "
            f"but carries {len(payload)}"
        )
    return PreviewImage(width=width, height=height, nv12=payload[:expected])
