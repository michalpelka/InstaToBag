"""Parsing of the Insta360 lens calibration strings found in the metadata record.

The camera stores calibration as underscore-joined decimal text.  Two variants matter:

``offset`` (metadata field 5/17), one group of 6 values per lens::

    <n>_<f>_<cx>_<cy>_<rx>_<ry>_<rz>_ ... _<canvas_w>_<canvas_h>_<crop>

``offset_v2`` (metadata field 54/56), one group of 19 values per lens, which is the
useful one because it carries separate fx/fy plus the lens model's parameters::

    <n>_ [ <xi> <fx> <fy> <cx> <cy> <rx> <ry> <rz> <tx> <ty> <tz>
           <k1> <k2> <k3> <p1> <p2> <canvas_w> <canvas_h> <crop> ] * n _<trailer>

All pixel coordinates are expressed on a *stitched dual-fisheye canvas* of
``canvas_w x canvas_h`` in which each lens occupies a ``canvas_w/2`` wide square, so
lens ``i`` has its principal point offset by ``i * canvas_w/2``.  Frames stored in the
file are usually smaller than that canvas (2880x2880 against a 5376-wide half on an X5),
so the intrinsics are rescaled to the actual frame size on the way out.

The lens model is the unified (Mei) omnidirectional one, OpenCV's ``omnidir``: a ray is
put on the unit sphere, projected from a centre ``xi`` behind it, then given radial
(k1..k3) and tangential (p1, p2) distortion.  That reading matches Gyroflow's Insta360
model field for field, and it is the one that makes the numbers physical: with xi = 2 a
ray 100 deg off the axis lands on the rim of the image circle, as a ~200 deg lens should.

ROS has no such model -- sensor_msgs defines ``plumb_bob``, ``rational_polynomial`` and
``equidistant`` -- so CameraInfo carries an ``equidistant`` (Kannala-Brandt, as in
``cv::fisheye``) model fitted to the unified one over the whole image circle.  Radially
the two agree to a fraction of a pixel; the tangential terms have no equidistant
counterpart and are dropped, which costs a few pixels at the rim.  The fit's worst error
is kept on each lens, and the camera's own values travel verbatim on the metadata topic.

The same lens model is what HDMapping writes out after calibrating the camera against a
lidar, so :func:`build_lens` takes unified-model intrinsics from either source and does
the rescaling and the fit once; see :mod:`insta360_to_bag.hdmapping`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

#: The fisheye model sensor_msgs defines: Kannala-Brandt, as used by cv::fisheye.
DISTORTION_MODEL = "equidistant"

#: Where a lens calibration came from, reported on the metadata topic and by --inspect.
SOURCE_METADATA = "insta360 metadata offset_v2"

_V2_FIELDS_PER_LENS = 19

#: Rays sampled along the radius for the least-squares fit.
_FIT_SAMPLES = 400
#: Radius x azimuth grid on which the fit is checked against the full unified model.
_CHECK_RADII = 200
_CHECK_AZIMUTHS = 72


@dataclass(frozen=True)
class LensCalibration:
    """One fisheye lens, rescaled to the stored frame size.

    Holds the camera's own unified-model calibration together with the equidistant
    model fitted to it, which is what CameraInfo publishes.
    """

    index: int
    width: int
    height: int
    #: Unified-model intrinsics.  fx is not a pinhole focal length: near the axis a ray
    #: at angle theta lands fx * theta / (1 + xi) pixels from the principal point.
    fx: float
    fy: float
    cx: float
    cy: float
    #: Distance of the unified model's projection centre behind the unit sphere.
    xi: float
    #: Unified-model k1, k2, k3, p1, p2, verbatim from the camera.
    distortion: List[float]
    #: Lens yaw/pitch/roll in degrees, as reported by the camera.
    rotation_deg: Tuple[float, float, float]
    #: Lens translation as reported by the camera: apparently metres, second lens
    #: relative to the first.
    translation: Tuple[float, float, float]
    #: Image the raw values were expressed on -- the stitched canvas for metadata
    #: calibration, the calibrated image size for an external one -- kept for traceability.
    canvas: Tuple[int, int]
    #: Horizontal and vertical rescaling applied to get from that image to the frame.
    scale: Tuple[float, float]
    #: Equidistant (Kannala-Brandt) k1..k4 fitted to the unified model.
    equidistant: List[float]
    #: Angle from the optical axis out to which the fit was made and checked, degrees.
    fit_max_angle_deg: float
    #: Worst distance between the equidistant and the full unified projection within
    #: that angle, in pixels of this frame size.
    fit_error_px: float
    #: Where these values came from: :data:`SOURCE_METADATA`, or an external
    #: calibration that replaced it.
    source: str = SOURCE_METADATA

    @property
    def equidistant_fx(self) -> float:
        return self.fx / (1.0 + self.xi)

    @property
    def equidistant_fy(self) -> float:
        return self.fy / (1.0 + self.xi)

    @property
    def k(self) -> List[float]:
        """Camera matrix of the published equidistant model."""
        return [
            self.equidistant_fx, 0.0, self.cx,
            0.0, self.equidistant_fy, self.cy,
            0.0, 0.0, 1.0,
        ]

    @property
    def r(self) -> List[float]:
        # No stereo rectification is defined for a back-to-back fisheye pair.
        return [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

    @property
    def p(self) -> List[float]:
        return [
            self.equidistant_fx, 0.0, self.cx, 0.0,
            0.0, self.equidistant_fy, self.cy, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]


def _unified_radius(theta: float, xi: float, k1: float, k2: float, k3: float) -> float:
    """Radius on the unified model's normalised plane of a ray at ``theta``, radial terms only."""
    rho = math.sin(theta) / (math.cos(theta) + xi)
    r2 = rho * rho
    return rho * (1.0 + r2 * (k1 + r2 * (k2 + r2 * k3)))


def _unified_point(
    theta: float, phi: float, xi: float, distortion: List[float]
) -> Tuple[float, float]:
    """Full unified projection onto the normalised plane, tangential terms included."""
    k1, k2, k3, p1, p2 = distortion
    rho = math.sin(theta) / (math.cos(theta) + xi)
    x, y = rho * math.cos(phi), rho * math.sin(phi)
    r2 = x * x + y * y
    radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
    return (
        x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x),
        y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y,
    )


def _equidistant_angle(theta: float, coefficients: List[float]) -> float:
    k1, k2, k3, k4 = coefficients
    t2 = theta * theta
    return theta * (1.0 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))


def _solve(matrix: List[List[float]], vector: List[float]) -> List[float]:
    """Gaussian elimination with partial pivoting, for the 4x4 normal equations below."""
    n = len(vector)
    rows = [list(matrix[i]) + [vector[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(rows[row][col]))
        if rows[pivot][col] == 0.0:
            raise ValueError("equidistant fit is singular")
        rows[col], rows[pivot] = rows[pivot], rows[col]
        for row in range(col + 1, n):
            factor = rows[row][col] / rows[col][col]
            rows[row] = [a - factor * b for a, b in zip(rows[row], rows[col])]
    solution = [0.0] * n
    for i in reversed(range(n)):
        tail = sum(rows[i][j] * solution[j] for j in range(i + 1, n))
        solution[i] = (rows[i][n] - tail) / rows[i][i]
    return solution


def _fit_equidistant(
    xi: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    distortion: List[float],
) -> Tuple[List[float], float, float]:
    """Fit Kannala-Brandt k1..k4 to a unified-model lens.

    Returns the coefficients, the largest angle fitted (degrees) and the worst pixel
    error against the full unified model inside it.  With a focal length of
    ``f / (1 + xi)`` both models agree to first order at the axis, leaving only the
    shape of the radial curve to fit: ``theta_d(theta) = (1 + xi) * r_unified(theta)``.
    The fit is least squares in radius, i.e. in pixels, out to where the image circle
    reaches the farthest frame edge, or to where the unified model folds back on itself
    if that comes first.
    """
    k1, k2, k3 = distortion[:3]

    def radius(theta: float) -> float:
        return _unified_radius(theta, xi, k1, k2, k3)

    step = math.radians(0.1)
    peak = step
    while peak + step < math.pi and radius(peak + step) > radius(peak):
        peak += step

    # Normalised radius of the frame edge farthest from the principal point.
    reach = max(max(cx, width - cx) / fx, max(cy, height - cy) / fy)
    limit = peak
    if radius(peak) > reach:
        low, high = 0.0, peak
        for _ in range(60):
            middle = 0.5 * (low + high)
            if radius(middle) < reach:
                low = middle
            else:
                high = middle
        limit = high

    gram = [[0.0] * 4 for _ in range(4)]
    rhs = [0.0] * 4
    for i in range(1, _FIT_SAMPLES + 1):
        theta = limit * i / _FIT_SAMPLES
        target = (1.0 + xi) * radius(theta) - theta
        basis = [theta ** (2 * j + 3) for j in range(4)]
        for a in range(4):
            rhs[a] += basis[a] * target
            for b in range(4):
                gram[a][b] += basis[a] * basis[b]
    coefficients = _solve(gram, rhs)

    fx_equidistant, fy_equidistant = fx / (1.0 + xi), fy / (1.0 + xi)
    worst = 0.0
    for i in range(1, _CHECK_RADII + 1):
        theta = limit * i / _CHECK_RADII
        theta_d = _equidistant_angle(theta, coefficients)
        for j in range(_CHECK_AZIMUTHS):
            phi = 2.0 * math.pi * j / _CHECK_AZIMUTHS
            x, y = _unified_point(theta, phi, xi, distortion)
            worst = max(
                worst,
                math.hypot(
                    fx * x - fx_equidistant * theta_d * math.cos(phi),
                    fy * y - fy_equidistant * theta_d * math.sin(phi),
                ),
            )
    return coefficients, math.degrees(limit), worst


def build_lens(
    *,
    index: int,
    xi: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    distortion: List[float],
    source_size: Tuple[float, float],
    width: int,
    height: int,
    canvas: Optional[Tuple[int, int]] = None,
    rotation_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    translation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    source: str = SOURCE_METADATA,
) -> Optional[LensCalibration]:
    """Rescale one unified-model lens onto ``width`` x ``height`` and fit an equidistant model.

    ``source_size`` is the image the raw intrinsics were measured on: half the stitched
    canvas for the camera's own calibration, the calibrated image size for an external
    one.  Only fx, fy, cx and cy are scaled -- the distortion coefficients live on the
    normalised plane and are size-independent.

    Returns ``None`` if the values cannot make a usable lens, so that every caller
    degrades to "no CameraInfo" rather than to silently wrong intrinsics.
    """
    source_width, source_height = source_size
    if source_width <= 0 or source_height <= 0 or fx <= 0 or fy <= 0 or xi <= -1.0:
        return None
    if len(distortion) != 5:
        return None

    # Scale each axis independently so that a non-square output (from --scale, or a
    # calibration measured at a different aspect ratio) still gets consistent intrinsics.
    scale_x = width / float(source_width)
    scale_y = height / float(source_height)
    fx, fy = fx * scale_x, fy * scale_y
    cx, cy = cx * scale_x, cy * scale_y
    try:
        equidistant, fit_angle, fit_error = _fit_equidistant(
            xi, fx, fy, cx, cy, width, height, distortion
        )
    except (ValueError, ZeroDivisionError, OverflowError):
        return None

    return LensCalibration(
        index=index,
        width=width,
        height=height,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        xi=xi,
        distortion=list(distortion),
        rotation_deg=rotation_deg,
        translation=translation,
        canvas=canvas or (int(source_width), int(source_height)),
        scale=(scale_x, scale_y),
        equidistant=equidistant,
        fit_max_angle_deg=fit_angle,
        fit_error_px=fit_error,
        source=source,
    )


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
            xi = float(group[0])
            fx, fy, cx, cy = (float(v) for v in group[1:5])
            rx, ry, rz = (float(v) for v in group[5:8])
            tx, ty, tz = (float(v) for v in group[8:11])
            distortion = [float(v) for v in group[11:16]]
            canvas_w, canvas_h = int(float(group[16])), int(float(group[17]))
        except ValueError:
            return []
        if canvas_w <= 0 or canvas_h <= 0:
            return []

        # Each lens occupies the left or right square half of the stitched canvas, so
        # its principal point has to be brought back from that half's origin.
        half_width = canvas_w / 2.0
        lens = build_lens(
            index=index,
            xi=xi,
            fx=fx,
            fy=fy,
            cx=cx - index * half_width,
            cy=cy,
            distortion=distortion,
            source_size=(half_width, float(canvas_h)),
            width=width,
            height=height,
            canvas=(canvas_w, canvas_h),
            rotation_deg=(rx, ry, rz),
            translation=(tx, ty, tz),
        )
        if lens is None:
            return []
        lenses.append(lens)
    return lenses


def summarise(lens: LensCalibration, label: Optional[str] = None) -> str:
    """One line describing a lens.  ``label`` replaces the default ``lens <index>``."""
    return (
        f"{label or f'lens {lens.index}'}: unified xi={lens.xi:g} "
        f"fx={lens.fx:.2f} fy={lens.fy:.2f} "
        f"cx={lens.cx:.2f} cy={lens.cy:.2f} "
        f"(rescaled from {lens.canvas[0]}x{lens.canvas[1]} "
        f"by {lens.scale[0]:.5f}x{lens.scale[1]:.5f})"
    )
