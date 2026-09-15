"""Reading the camera calibration files written by HDMapping.

HDMapping (https://github.com/MapsHD/HDMapping) calibrates a camera against the lidar it
is mounted with and writes one JSON file per camera::

    {
      "extrinsics": {
        "T_lidar_to_camera_4x4": [[...], [...], [...], [0, 0, 0, 1]],
        "camera_position_in_world_xyz": [x, y, z],
        "camera_rotation_matrix_in_world": [[...], [...], [...]]
      },
      "serial": "...", "model": "...", "firmware": "...",
      "intrinsics": {"model": "mei", "xi": ..., "fx": ..., "fy": ..., "cx": ..., "cy": ...,
                     "k1": ..., "k2": ..., "k3": ..., "k4": ..., "k5": ..., "k6": ...,
                     "p1": ..., "p2": ..., "width": ..., "height": ...}
    }

Two things make this worth embedding.  The extrinsics are *measured* rather than the
nominal mount ``--camera-xyz``/``--camera-rpy`` describe, and they are per lens, so the
few centimetres between each lens's optical centre and the body centre are carried too.

``T_lidar_to_camera_4x4`` takes a point from the lidar frame into the camera frame, so
the transform ROS wants -- the camera's pose *in* the lidar frame, which is what a
``TransformStamped`` from the lidar frame to the camera's frame holds -- is its inverse:
``R_pose = R^T`` and ``t_pose = -R^T t``.  The other two extrinsic fields are that
inverse already, and are cross-checked against the 4x4 rather than trusted blindly.

The camera frame is the optical one ROS uses for images: z along the view, x to image
right, y to image down.  That was established by comparing a calibration of the rig this
tool was developed on against the nominal mount it documents -- the two agree to within
about three degrees per lens, which is the size of a mount tolerance and not of a
convention difference.

The intrinsics are the same unified (Mei) model the camera itself reports, expressed on
the calibrated image size, so they go through the same rescaling and equidistant fit as
the camera's own values; see :mod:`insta360_to_bag.calibration`.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .calibration import LensCalibration, build_lens

#: The only intrinsics model this reader understands.  Anything else names coefficients
#: that mean something different, so it is refused rather than misread.
SUPPORTED_MODEL = "mei"

#: Largest departure from orthonormality tolerated in the rotation block, and the point
#: past which it is no longer a rounding artefact of a 32-bit float dump.
_ROTATION_WARN = 1e-4
_ROTATION_FAIL = 1e-2

#: Tolerance for the file's redundant pose fields against the 4x4 they duplicate.
_CONSISTENCY_TOLERANCE = 1e-5


class CalibrationError(ValueError):
    """An HDMapping calibration file that cannot be read as one."""


@dataclass(frozen=True)
class Pose:
    """A camera optical frame in the lidar frame: metres, and an (x, y, z, w) quaternion."""

    translation: Tuple[float, float, float]
    rotation: Tuple[float, float, float, float]


@dataclass(frozen=True)
class CameraCalibration:
    """One camera's calibration, as read from a single HDMapping JSON file."""

    path: str
    #: Measured pose of the camera's optical frame in the lidar frame.
    pose: Pose
    #: Unified-model intrinsics, on the image size they were calibrated at.
    width: int
    height: int
    xi: float
    fx: float
    fy: float
    cx: float
    cy: float
    #: k1, k2, k3, p1, p2, in the order :class:`~insta360_to_bag.calibration.LensCalibration` wants.
    distortion: List[float]
    #: Camera identification as the file records it, for cross-checking against the capture.
    serial: Optional[str] = None
    model: Optional[str] = None
    firmware: Optional[str] = None
    #: Non-fatal problems found while reading, to be surfaced by the caller.
    warnings: List[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        """The file's stem, which is how HDMapping names the camera (``cam_front``)."""
        return os.path.splitext(os.path.basename(self.path))[0]

    def lens(self, index: int, width: int, height: int) -> Optional[LensCalibration]:
        """Rescale onto a ``width`` x ``height`` frame, as :func:`.calibration.build_lens`."""
        return build_lens(
            index=index,
            xi=self.xi,
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            distortion=self.distortion,
            source_size=(float(self.width), float(self.height)),
            width=width,
            height=height,
            canvas=(self.width, self.height),
            source=f"hdmapping {self.path}",
        )


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalibrationError(f"{where} is not a number")
    result = float(value)
    if not math.isfinite(result):
        raise CalibrationError(f"{where} is not finite")
    return result


def _mapping(value: Any, where: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise CalibrationError(f"{where} is missing or is not an object")
    return value


def _matrix(value: Any, rows: int, columns: int, where: str) -> List[List[float]]:
    if not isinstance(value, list) or len(value) != rows:
        raise CalibrationError(f"{where} is not a {rows}x{columns} matrix")
    out: List[List[float]] = []
    for r, row in enumerate(value):
        if not isinstance(row, list) or len(row) != columns:
            raise CalibrationError(f"{where} row {r} is not {columns} values long")
        out.append([_number(cell, f"{where}[{r}][{c}]") for c, cell in enumerate(row)])
    return out


def _quaternion_from_matrix(
    m: Sequence[Sequence[float]],
) -> Tuple[float, float, float, float]:
    """Rotation matrix to a unit (x, y, z, w) quaternion, by Shepperd's branch on the trace.

    Branching on the largest diagonal term keeps the divisor away from zero, which the
    bare trace formula cannot do for rotations near 180 degrees -- and every lens here
    sits within a few degrees of one.

    The result is normalised: these files hold a 32-bit float dump, whose rotation is
    orthonormal only to about 1e-8, and tf2 wants a unit quaternion.
    """
    quaternion = _quaternion_from_matrix_raw(m)
    norm = math.sqrt(math.fsum(c * c for c in quaternion))
    if norm == 0.0:
        raise CalibrationError("the rotation does not give a usable quaternion")
    return tuple(c / norm for c in quaternion)  # type: ignore[return-value]


def _quaternion_from_matrix_raw(
    m: Sequence[Sequence[float]],
) -> Tuple[float, float, float, float]:
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return ((m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s,
                (m[1][0] - m[0][1]) / s, 0.25 * s)
    if m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
        return (0.25 * s, (m[0][1] + m[1][0]) / s,
                (m[0][2] + m[2][0]) / s, (m[2][1] - m[1][2]) / s)
    if m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
        return ((m[0][1] + m[1][0]) / s, 0.25 * s,
                (m[1][2] + m[2][1]) / s, (m[0][2] - m[2][0]) / s)
    s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
    return ((m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s,
            0.25 * s, (m[1][0] - m[0][1]) / s)


def _orthonormality_error(rotation: Sequence[Sequence[float]]) -> float:
    """Largest deviation of ``R R^T`` from the identity."""
    worst = 0.0
    for i in range(3):
        for j in range(3):
            dot = sum(rotation[i][k] * rotation[j][k] for k in range(3))
            worst = max(worst, abs(dot - (1.0 if i == j else 0.0)))
    return worst


def _determinant(m: Sequence[Sequence[float]]) -> float:
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def _read_pose(extrinsics: Dict[str, Any], warnings: List[str]) -> Pose:
    matrix = _matrix(
        extrinsics.get("T_lidar_to_camera_4x4"), 4, 4, "T_lidar_to_camera_4x4"
    )
    rotation = [row[:3] for row in matrix[:3]]
    translation = [row[3] for row in matrix[:3]]

    error = _orthonormality_error(rotation)
    if error > _ROTATION_FAIL:
        raise CalibrationError(
            f"the rotation block of T_lidar_to_camera_4x4 is not a rotation "
            f"(off orthonormal by {error:.3g})"
        )
    if error > _ROTATION_WARN:
        warnings.append(
            f"rotation block is off orthonormal by {error:.3g}; used as given"
        )
    if _determinant(rotation) < 0.0:
        raise CalibrationError(
            "the rotation block of T_lidar_to_camera_4x4 has a negative determinant, "
            "so it mirrors rather than rotates"
        )
    if matrix[3] != [0.0, 0.0, 0.0, 1.0]:
        warnings.append(
            f"bottom row of T_lidar_to_camera_4x4 is {matrix[3]}, not [0, 0, 0, 1]; "
            "only its rotation and translation blocks are used"
        )

    # The pose is the inverse of the lidar-to-camera transform: R^T and -R^T t.
    inverse = [[rotation[j][i] for j in range(3)] for i in range(3)]
    position = tuple(
        -sum(inverse[i][k] * translation[k] for k in range(3)) for i in range(3)
    )

    # The file states that inverse a second time; disagreement means the two halves were
    # written from different states, so say so rather than silently pick one.
    stated_position = extrinsics.get("camera_position_in_world_xyz")
    if isinstance(stated_position, list) and len(stated_position) == 3:
        values = [_number(v, "camera_position_in_world_xyz") for v in stated_position]
        if any(abs(a - b) > _CONSISTENCY_TOLERANCE for a, b in zip(values, position)):
            warnings.append(
                f"camera_position_in_world_xyz {values} disagrees with the position "
                f"implied by T_lidar_to_camera_4x4 {[round(v, 9) for v in position]}; "
                "the 4x4 is used"
            )
    stated_rotation = extrinsics.get("camera_rotation_matrix_in_world")
    if isinstance(stated_rotation, list) and len(stated_rotation) == 3:
        values = _matrix(stated_rotation, 3, 3, "camera_rotation_matrix_in_world")
        if any(
            abs(values[i][j] - inverse[i][j]) > _CONSISTENCY_TOLERANCE
            for i in range(3)
            for j in range(3)
        ):
            warnings.append(
                "camera_rotation_matrix_in_world disagrees with the transpose of the "
                "rotation in T_lidar_to_camera_4x4; the 4x4 is used"
            )

    return Pose(translation=position, rotation=_quaternion_from_matrix(inverse))


def _read_intrinsics(
    intrinsics: Dict[str, Any], warnings: List[str]
) -> Dict[str, Any]:
    model = intrinsics.get("model")
    if not isinstance(model, str) or model.strip().lower() != SUPPORTED_MODEL:
        raise CalibrationError(
            f"intrinsics model is {model!r}; only {SUPPORTED_MODEL!r} (the unified "
            "omnidirectional model this camera uses) can be read"
        )

    values = {
        key: _number(intrinsics.get(key), f"intrinsics.{key}")
        for key in ("xi", "fx", "fy", "cx", "cy", "width", "height")
    }
    width, height = int(values["width"]), int(values["height"])
    if width <= 0 or height <= 0:
        raise CalibrationError(
            f"intrinsics give a {width}x{height} calibrated image size"
        )
    if values["fx"] <= 0 or values["fy"] <= 0:
        raise CalibrationError("intrinsics give a non-positive focal length")
    if values["xi"] <= -1.0:
        raise CalibrationError(f"intrinsics give xi = {values['xi']}, which is <= -1")

    distortion = [
        _number(intrinsics.get(key, 0.0), f"intrinsics.{key}")
        for key in ("k1", "k2", "k3", "p1", "p2")
    ]
    # The unified model here carries three radial terms; higher ones have nowhere to go.
    extra = {
        key: _number(intrinsics[key], f"intrinsics.{key}")
        for key in ("k4", "k5", "k6")
        if key in intrinsics
    }
    nonzero = {key: value for key, value in extra.items() if value != 0.0}
    if nonzero:
        warnings.append(
            "intrinsics carry non-zero "
            + ", ".join(f"{key}={value:g}" for key, value in sorted(nonzero.items()))
            + ", which the three-term radial model has no place for; they are dropped"
        )

    return {
        "width": width,
        "height": height,
        "xi": values["xi"],
        "fx": values["fx"],
        "fy": values["fy"],
        "cx": values["cx"],
        "cy": values["cy"],
        "distortion": distortion,
    }


def load(path: str) -> CameraCalibration:
    """Read an HDMapping calibration file.

    Raises :class:`CalibrationError` when the file cannot be read as one.  Problems that
    leave the numbers usable -- a redundant field disagreeing, distortion terms with no
    home -- are collected in :attr:`CameraCalibration.warnings` instead, so that they
    reach the bag's warning list rather than stopping a conversion.
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            document = json.load(handle)
    except OSError as exc:
        raise CalibrationError(f"could not read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise CalibrationError(f"{path} is not valid JSON: {exc}") from exc

    if not isinstance(document, dict):
        raise CalibrationError(f"{path} does not hold a JSON object")

    warnings: List[str] = []
    try:
        pose = _read_pose(_mapping(document.get("extrinsics"), "extrinsics"), warnings)
        intrinsics = _read_intrinsics(
            _mapping(document.get("intrinsics"), "intrinsics"), warnings
        )
    except CalibrationError as exc:
        raise CalibrationError(f"{path}: {exc}") from exc

    def text(key: str) -> Optional[str]:
        value = document.get(key)
        return value if isinstance(value, str) and value else None

    return CameraCalibration(
        path=path,
        pose=pose,
        serial=text("serial"),
        model=text("model"),
        firmware=text("firmware"),
        warnings=warnings,
        **intrinsics,
    )
