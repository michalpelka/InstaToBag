"""ROS 2 message definitions, embedded as text.

MCAP stores each channel's schema inside the bag, so the definitions live here rather
than being imported from an installed ROS distribution.  That keeps the converter
runnable on a machine with no ROS at all, and makes the resulting bag fully
self-describing: ``ros2 bag`` and Foxglove both read the schema out of the file.

The ``sensor_msgs``, ``std_msgs``, ``geometry_msgs``, ``tf2_msgs`` and
``builtin_interfaces`` definitions below are verbatim copies of the upstream ``.msg`` files (comments
stripped, dependency type names fully qualified).  ``audio_common_msgs/msg/AudioData``
is the long-standing definition from ros-drivers/audio_common; since the schema
travels inside the bag, a reader does not need that package installed either.
"""

from __future__ import annotations

_SEPARATOR = "=" * 80

_TIME = """MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
"""

_HEADER = """MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
"""

_VECTOR3 = """MSG: geometry_msgs/Vector3
float64 x
float64 y
float64 z
"""

_QUATERNION = """MSG: geometry_msgs/Quaternion
float64 x 0
float64 y 0
float64 z 0
float64 w 1
"""

_REGION_OF_INTEREST = """MSG: sensor_msgs/RegionOfInterest
uint32 x_offset
uint32 y_offset
uint32 height
uint32 width
bool do_rectify
"""

_TRANSFORM_STAMPED = """MSG: geometry_msgs/TransformStamped
std_msgs/Header header
string child_frame_id
geometry_msgs/Transform transform
"""

_TRANSFORM = """MSG: geometry_msgs/Transform
geometry_msgs/Vector3 translation
geometry_msgs/Quaternion rotation
"""


def _concat(root: str, *dependencies: str) -> str:
    parts = [root]
    for dependency in dependencies:
        parts.append(f"{_SEPARATOR}\n{dependency}")
    return "".join(parts)


COMPRESSED_IMAGE = "sensor_msgs/msg/CompressedImage"
IMAGE = "sensor_msgs/msg/Image"
CAMERA_INFO = "sensor_msgs/msg/CameraInfo"
IMU = "sensor_msgs/msg/Imu"
FLOAT64 = "std_msgs/msg/Float64"
STRING = "std_msgs/msg/String"
AUDIO_DATA = "audio_common_msgs/msg/AudioData"
TF_MESSAGE = "tf2_msgs/msg/TFMessage"


DEFINITIONS = {
    COMPRESSED_IMAGE: _concat(
        """std_msgs/Header header
string format
uint8[] data
""",
        _HEADER,
        _TIME,
    ),
    IMAGE: _concat(
        """std_msgs/Header header
uint32 height
uint32 width
string encoding
uint8 is_bigendian
uint32 step
uint8[] data
""",
        _HEADER,
        _TIME,
    ),
    CAMERA_INFO: _concat(
        """std_msgs/Header header
uint32 height
uint32 width
string distortion_model
float64[] d
float64[9] k
float64[9] r
float64[12] p
uint32 binning_x
uint32 binning_y
sensor_msgs/RegionOfInterest roi
""",
        _HEADER,
        _TIME,
        _REGION_OF_INTEREST,
    ),
    IMU: _concat(
        """std_msgs/Header header
geometry_msgs/Quaternion orientation
float64[9] orientation_covariance
geometry_msgs/Vector3 angular_velocity
float64[9] angular_velocity_covariance
geometry_msgs/Vector3 linear_acceleration
float64[9] linear_acceleration_covariance
""",
        _HEADER,
        _TIME,
        _QUATERNION,
        _VECTOR3,
    ),
    FLOAT64: "float64 data\n",
    STRING: "string data\n",
    AUDIO_DATA: "uint8[] data\n",
    TF_MESSAGE: _concat(
        "geometry_msgs/TransformStamped[] transforms\n",
        _TRANSFORM_STAMPED,
        _HEADER,
        _TIME,
        _TRANSFORM,
        _VECTOR3,
        _QUATERNION,
    ),
}
