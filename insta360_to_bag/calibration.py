"""Parsing of the Insta360 lens calibration strings found in the metadata record.

The camera stores calibration as underscore-joined decimal text.  Two variants matter:

``offset`` (metadata field 5/17), one group of 6 values per lens::

    <n>_<f>_<cx>_<cy>_<rx>_<ry>_<rz>_ ... _<canvas_w>_<canvas_h>_<crop>

``offset_v2`` (metadata field 54/56), one group of 19 values per lens, which is the
useful one because it carries separate fx/fy plus a distortion polynomial::

    <n>_ [ <type> <fx> <fy> <cx> <cy> <rx> <ry> <rz> <tx> <ty> <tz>
           <k1> <k2> <k3> <k4> <k5> <canvas_w> <canvas_h> <crop> ] * n _<trailer>

All pixel coordinates are expressed on a *stitched dual-fisheye canvas* of
``canvas_w x canvas_h`` in which each lens occupies a ``canvas_w/2`` wide square, so
lens ``i`` has its principal point offset by ``i * canvas_w/2``.  Frames stored in the
file are usually smaller than that canvas (2880x2880 against a 5376-wide half on an X5),
so the intrinsics are rescaled to the actual frame size on the way out.

The distortion coefficients are Insta360's own polynomial, *not* OpenCV's ``plumb_bob``
or ``equidistant``.  They are passed through verbatim under a distortion model name of
``insta360_fisheye_v2`` so that no consumer mistakes them for a standard model, and the
raw strings are republished untouched so nothing is lost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

DISTORTION_MODEL = "insta360_fisheye_v2"

_V2_FIELDS_PER_LENS = 19


@dataclass(frozen=True)
class LensCalibration:
    """Intrinsics for one fisheye lens, rescaled to the stored frame size."""

    index: int
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: List[float]
    #: Lens yaw/pitch/roll in degrees, as reported by the camera.
    rotation_deg: tuple[float, float, float]
    #: Lens translation, in canvas pixels, as reported by the camera.
    translation: tuple[float, float, float]
    #: Canvas the raw values were expressed on, kept for traceability.
    canvas: tuple[int, int]
    #: Horizontal and vertical rescaling applied to get from that canvas to the frame.
    scale: tuple[float, float]

    @property
    def k(self) -> List[float]:
        return [self.fx, 0.0, self.cx, 0.0, self.fy, self.cy, 0.0, 0.0, 1.0]

    @property
    def r(self) -> List[float]:
        # No stereo rectification is defined for a back-to-back fisheye pair.
        return [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

    @property
    def p(self) -> List[float]:
        return [
            self.fx, 0.0, self.cx, 0.0,
            0.0, self.fy, self.cy, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]


def parse_offset_v2(text: str, width: int, height: int) -> List[LensCalibration]:
    """Parse a field-54/56 calibration string, rescaled to ``width`` x ``height``.

    Returns an empty list if the string does not have the expected shape, so that a
    firmware revision with a different layout degrades to "no CameraInfo" rather than
    to silently wrong intrinsics.
    """
    tokens = text.split("_")
    if len(tokens) < 2:
        return []
    try:
        lens_count = int(tokens[0])
    except ValueError:
        return []
    body = tokens[1:]
    # A single trailing value (a byte offset into the file) follows the lens groups.
    if lens_count <= 0 or len(body) < lens_count * _V2_FIELDS_PER_LENS:
        return []

    lenses: List[LensCalibration] = []
    for index in range(lens_count):
        group = body[index * _V2_FIELDS_PER_LENS : (index + 1) * _V2_FIELDS_PER_LENS]
        try:
            fx, fy, cx, cy = (float(v) for v in group[1:5])
            rx, ry, rz = (float(v) for v in group[5:8])
            tx, ty, tz = (float(v) for v in group[8:11])
            distortion = [float(v) for v in group[11:16]]
            canvas_w, canvas_h = int(float(group[16])), int(float(group[17]))
        except ValueError:
            return []
        if canvas_w <= 0 or canvas_h <= 0:
            return []

        # Each lens occupies the left or right square half of the stitched canvas.
        half_width = canvas_w / 2.0
        # Scale each axis independently so that a non-square output (from --scale)
        # still gets consistent intrinsics.
        scale_x = width / half_width
        scale_y = height / float(canvas_h)
        lenses.append(
            LensCalibration(
                index=index,
                width=width,
                height=height,
                fx=fx * scale_x,
                fy=fy * scale_y,
                cx=(cx - index * half_width) * scale_x,
                cy=cy * scale_y,
                distortion=distortion,
                rotation_deg=(rx, ry, rz),
                translation=(tx, ty, tz),
                canvas=(canvas_w, canvas_h),
                scale=(scale_x, scale_y),
            )
        )
    return lenses


def summarise(lens: LensCalibration) -> str:
    return (
        f"lens {lens.index}: fx={lens.fx:.2f} fy={lens.fy:.2f} "
        f"cx={lens.cx:.2f} cy={lens.cy:.2f} "
        f"(rescaled from a {lens.canvas[0]}x{lens.canvas[1]} canvas "
        f"by {lens.scale[0]:.5f}x{lens.scale[1]:.5f})"
    )
