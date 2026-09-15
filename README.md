# insta360-to-bag

Convert an Insta360 `.insv` capture into a ROS 2 MCAP bag: both fisheye tracks, the
1 kHz IMU, per-lens intrinsics, per-frame exposure, the embedded preview image and the
audio track, all on a single time base recovered from the camera's own clock.

Developed and verified against an **Insta360 X5** (firmware `v1.10.11_build1`, 5.7K
dual-fisheye, 2880×2880 per lens at 24 fps). The trailer format is shared across the
X-series, so ONE X2/X3/X4 files should work; the parsers degrade to "field not present"
rather than to wrong values when a firmware revision differs.

## Install


From a checkout, drop the PyPI name for a path so pipx builds from source instead
(add `--editable` to have it track the checkout live, so local edits take effect
without reinstalling):

```bash
pipx install --editable .
```

Or from a checkout, for development:

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'
```

`ffmpeg` and `ffprobe` must be on `PATH` (`sudo apt install ffmpeg`) — pip cannot
install those, and the converter exits with a clear message if they are missing. No ROS
installation is needed — message schemas are embedded in the bag, so the converter
runs anywhere and the resulting file is self-describing.

## Use

```bash
# See what's in a file without writing anything
insta360-to-bag VID_20260909_132827_00_005.insv --inspect

# Convert (writes VID_20260909_132827_00_005.mcap beside the input)
insta360-to-bag VID_20260909_132827_00_005.insv

# A quick look at a long clip, at a quarter resolution
insta360-to-bag capture.insv -o small.mcap --max-frames 100 --scale 0.25

# IMU only
insta360-to-bag capture.insv --no-video --no-audio --no-preview

# With a measured lidar-to-camera calibration from HDMapping
insta360-to-bag capture.insv \
    --hdmapping-calibration cam_front.json --hdmapping-calibration cam_back.json
```

Run `insta360-to-bag --help` for the full set. The useful ones:

| Flag | Effect |
| --- | --- |
| `--jpeg-quality N` | ffmpeg MJPEG quality, 2 (best) to 31. Default 3. |
| `--scale SPEC` | `0.5` or `1440x1440`. Intrinsics are rescaled to match. |
| `--max-frames N` | Stop after N frames. |
| `--swap-lenses` | Map the second video track to `cam_front`. |
| `--no-{video,camera-info,imu,exposure,preview,audio,tf}` | Leave a stream out. |
| `--camera-xyz X Y Z` | Camera centre in the lidar frame, metres. Default `0 0 0`. |
| `--camera-rpy R P Y` | Camera body orientation in the lidar frame, URDF roll/pitch/yaw in degrees. Default `90 0 90`. |
| `--lidar-frame FRAME` | Parent frame of the camera transforms. Default `lidar`. |
| `--hdmapping-calibration [LENS=]PATH` | Use a measured HDMapping calibration for one lens. Repeatable. |
| `--relative-time` | Start timestamps at zero instead of the capture wall clock. |
| `--compression {zstd,lz4,none}` | MCAP chunk compression. Default `zstd`. |
| `--topic-prefix` | Default `/insta360`. |

## Topics

| Topic | Type | Rate |
| --- | --- | --- |
| `/insta360/cam_front/image/compressed` | `sensor_msgs/msg/CompressedImage` (jpeg) | 24 Hz |
| `/insta360/cam_back/image/compressed` | `sensor_msgs/msg/CompressedImage` (jpeg) | 24 Hz |
| `/insta360/cam_front/camera_info` | `sensor_msgs/msg/CameraInfo` | 24 Hz |
| `/insta360/cam_back/camera_info` | `sensor_msgs/msg/CameraInfo` | 24 Hz |
| `/insta360/imu` | `sensor_msgs/msg/Imu` | ~1003 Hz |
| `/insta360/exposure_time` | `std_msgs/msg/Float64` (seconds) | 24 Hz |
| `/insta360/preview/image` | `sensor_msgs/msg/Image` (rgb8 equirect) | once |
| `/insta360/audio` | `audio_common_msgs/msg/AudioData` (PCM s16le) | 24 Hz |
| `/insta360/metadata` | `std_msgs/msg/String` (JSON) | once |
| `/tf_static` | `tf2_msgs/msg/TFMessage` (transient-local QoS) | once |

Messages are written in strict time order, and each message's `header.stamp` equals its
MCAP log time. `/tf_static` is never put under `--topic-prefix`, because tf listeners
only subscribe to that exact name.

The camera's **serial number**, model, firmware and capture time are also written as an
MCAP metadata record named `insta360`, alongside the path of any calibration file used.
That sits in the file's summary section rather than in a message, so a reader can tie a
bag to the hardware — and so to a calibration made for that serial — without decoding
anything: `mcap info capture.mcap` prints it, and `reader.iter_metadata()` returns it.
The same values are on `/insta360/metadata` for consumers that only read topics.

A 15.6 s 5.7K clip becomes roughly 350 MB at the default JPEG quality; conversion is
bounded by HEVC decode of two 2880×2880 tracks.

### Frames

`insta360_cam_front_optical_frame`, `insta360_cam_back_optical_frame`, `insta360_imu`.

`/tf_static` carries one transform from the lidar frame (`--lidar-frame`, default
`lidar`, REP 103 x-forward / y-left / z-up) to each lens's optical frame (z along the
viewing direction, x to image right, y to image down), built from two parts:

- **Lens in body.** The body frame is REP 103 for the camera itself: x along
  `cam_front`'s view, y to its left, z out of the top (the lens end). The stored frames
  are upright relative to the body — image down points toward the camera's base.
- **Body in lidar**, set by `--camera-xyz X Y Z` (metres) and `--camera-rpy R P Y`
  (URDF fixed-axis roll/pitch/yaw, degrees). Both lenses sit at the body's position.

The default `--camera-rpy 90 0 90` is the rig this was developed on, which carries the
camera **on its side**: lens end toward the lidar's +x, **`cam_front` looking left**
and **`cam_back` looking right** — which is why its frames show the world sideways. For
an upright camera looking the same way, pass `--camera-rpy 0 0 90`. With the default:

| Child frame | Looks along | Image right | Image down | Rotation from the lidar frame (x, y, z, w) |
| --- | --- | --- | --- | --- |
| `insta360_cam_front_optical_frame` | +y (left) | −z (down) | −x (backward) | (−0.5, 0.5, 0.5, 0.5) |
| `insta360_cam_back_optical_frame` | −y (right) | +z (up) | −x (backward) | (0.5, −0.5, 0.5, 0.5) |

The calibration's own per-lens angles are not applied: they are expressed in Insta360's
stitching frame, whose conventions are not established (its `rz ≈ 90°` is not a
physical sensor roll). If the lenses are the wrong way round use `--swap-lenses`; to
publish your own transforms instead, pass `--no-tf`. `insta360_imu` has no transform — see the caveats below.

### Calibration from HDMapping

A nominal mount is a guess about where the camera sits. If you have calibrated the rig
with [HDMapping](https://github.com/MapsHD/HDMapping), pass its per-camera JSON instead
and the measured pose is published verbatim:

```bash
insta360-to-bag capture.insv \
    --hdmapping-calibration cam_front.json --hdmapping-calibration cam_back.json
```

HDMapping names each file after the camera it calibrated, so `cam_front.json` and
`cam_back.json` need no further ceremony. For a file named anything else, say which lens
it belongs to: `--hdmapping-calibration cam_back=2026-09-09-run3.json`.

Each file replaces two things for the lens it covers:

- **The pose.** `T_lidar_to_camera_4x4` takes a point from the lidar frame into the
  camera frame, so what `/tf_static` needs is its inverse, and that is what gets
  published. Its camera frame is the ROS optical one, which is why it drops straight in.
  `--camera-xyz` and `--camera-rpy` no longer apply to that lens. Because each lens is
  calibrated separately, **the two no longer share a position** — the few centimetres
  between each optical centre and the body centre come through.
- **The intrinsics.** HDMapping writes the same unified (Mei) model the camera reports,
  so the values go through the same rescaling to your output frame size and the same
  equidistant fit, and `CameraInfo` carries the calibrated numbers.

Calibrate one lens and leave the other out and only that lens changes; the other stays on
the nominal mount. `/insta360/metadata` records, per lens, which of the two it got and
from which file, and the file paths also go into the bag's `insta360` metadata record.

Checks that would otherwise pass silently are reported as warnings: a file made for a
different serial than the capture, a rotation block that is not quite a rotation, the
file's redundant `camera_position_in_world_xyz` disagreeing with the 4x4 it duplicates
(the 4x4 wins), and distortion terms the three-term radial model has no place for. A file
that cannot be read at all stops the conversion rather than quietly producing a bag
without the calibration you asked for. `--inspect` shows what each file says, including
the intrinsics rescaled to your frame size, before you convert anything.

One thing to watch: calibrations are applied **by lens name**, and `--swap-lenses`
changes which video track each name refers to. Using both together is warned about.

## Read it back

```bash
ros2 bag info capture.mcap
ros2 bag play capture.mcap
```

The bag also opens directly in Foxglove. Verified end to end: `ros2 bag info` reports
all nine topics, and every message deserializes against the real `sensor_msgs`
typesupport via `rosbag2_py`.

Reading a bag needs nothing installed — schemas travel inside the file. *Replaying* one
onto live topics is different: `ros2 bag play` has to build a publisher, which needs the
message type installed on the playback machine. Every topic here is a core type except
`/insta360/audio`, so without `audio_common_msgs` on the consumer side `ros2 bag play`
logs `Publisher for topic '/insta360/audio' not found` and plays the rest normally. Use
`--no-audio` at conversion time, or install `audio_common`, if that matters to you.

## Caveats — read before you use the data

These are the places where the camera does not tell us enough to be certain, and where
the tool deliberately preserves raw values instead of guessing.

- **IMU axes are the sensor's own, not REP 103.** Accelerometer and gyroscope are
  converted to m/s² and rad/s, but they are *not* rotated into ROS's x-forward /
  y-left / z-up convention, because the mapping from the X-series sensor frame to the
  camera body frame is undocumented. Determine it for your rig before fusing.
- **No IMU-to-camera extrinsics.** The metadata carries per-lens rotations, but not a
  calibrated IMU-to-camera transform, so `insta360_imu` is left out of `/tf_static`.
  The per-lens rotation and translation values are passed through on the metadata
  topic.
- **Lidar-to-camera transforms are nominal unless you supply a calibration.** By
  default the pose is whatever `--camera-xyz` and `--camera-rpy` say (the side mount
  described under Frames, at the lidar origin), and both lenses share one position — the
  few centimetres from each lens's optical centre to the body centre are not applied.
  Calibrate the rig if you need better than that, and pass the result with
  `--hdmapping-calibration`; the metadata topic says which of the two each lens got.
- **CameraInfo is an `equidistant` fit, not the camera's own model.** The camera
  calibrates each lens with the unified (Mei) omnidirectional model — OpenCV's
  `omnidir`: ξ = 2, radial k1–k3, tangential p1, p2 — which ROS does not define.
  `CameraInfo` therefore carries `distortion_model: equidistant` (Kannala–Brandt, what
  `cv::fisheye` and `image_proc` use), with `d` = k1–k4 fitted to the unified model and
  `K`/`P` built on the equivalent focal length fx / (1 + ξ). Radially the two agree to
  about 0.3 px out to the rim of the image circle; the tangential terms have no
  equidistant counterpart, and dropping them costs up to ~2 px on the front lens and
  ~4 px on the back one at the rim, at full resolution. `--inspect` prints the fit and
  its error for your file. The unified parameters are on `/insta360/metadata`, next to
  the raw calibration strings. A pinhole rectification can show at most the central
  < 180° of a ~200° lens. An HDMapping calibration uses the same model, so it is fitted
  the same way and carries the same caveat.
- **`cam_front` / `cam_back` is naming, not a determination.** They follow the order of
  the video tracks in the file, which is stable but has not been confirmed against
  which lens physically faces the screen. Use `--swap-lenses` if you need them the
  other way round — that also swaps which lens `/tf_static` puts on the left.
- **JPEG re-encode is lossy.** Frames are decoded from HEVC and re-encoded as JPEG.
  Lower `--jpeg-quality` for more fidelity, or keep the original file for archival.
- **IMU covariances are unknown.** They are left as zeros, which `sensor_msgs/Imu`
  defines as "unknown"; `orientation_covariance[0]` is `-1` to mark the identity
  orientation as absent, as the message documentation requires.

## The `.insv` format

An `.insv` is a plain MP4 — two fisheye HEVC tracks plus AAC audio, readable by any
tool — with a proprietary metadata trailer glued onto the end. Walking backwards from
EOF:

```
...MP4 boxes... | record | footer | record | footer | ... | directory | footer | tail

tail (72 bytes)   32 bytes reserved (zero)
                  uint32 le  trailer size in bytes, counted back from EOF
                  uint32 le  trailer format version (3 on current firmware)
                  char[32]   "8db42d694ccc418790edff439fe026bf"

footer (6 bytes)  uint16 be  record id      <- big-endian here
                  uint32 le  payload length

directory         10 reserved bytes, then 10-byte entries of
                  uint16 le  record id      <- little-endian here
                  uint32 le  length
                  uint32 le  offset, relative to the start of the trailer
                  plus zero-filled unused slots
```

The directory is what this tool reads; the per-record footers are cross-checked and any
disagreement is reported as a warning rather than treated as fatal.

Records seen on an X5:

| Id | Contents |
| --- | --- |
| `0x0101` | Protobuf metadata: serial, model, firmware, geometry, clock, calibration |
| `0x0002` | 1280×640 NV12 equirectangular preview image (40-byte header) |
| `0x0003` | IMU, `<Q6H` per sample at ~1 kHz |
| `0x0004` | Exposure, `<Qd` per captured frame |
| `0x0016` | Multi-frame preview sequence (not parsed) |
| `0x0009`, `0x000b` | Per-frame ISP statistics (not parsed) |
| `0x000a`, `0x001c`, `0x001d` | Sensor map, lookup table, frame marks (not parsed) |

### Timestamps

Everything in the trailer is stamped in microseconds since the camera booted. Metadata
field 98 pins one point of that clock — the first encoded video frame — to a UTC epoch
value, which is what lets the bag carry real wall-clock times. Frame timestamps come
from the exposure record, whose entries carry the sensor's true (slightly non-nominal)
frame interval and whose first in-range entry lands exactly on that pinned first-frame
timestamp; the nominal frame rate is only a fallback.

The IMU typically starts about a second before the first encoded frame, so the bag
begins earlier than the video does.

### How the IMU decoding was pinned down

Record `0x0003` is `<Q6H`: a timestamp then six 16-bit channels. Two things about it
are not self-evident, and both were settled by physics rather than by inspection:

- The counts are **offset binary** — zero sits at `0x8000`, not at 0. Read as two's
  complement, four of the six channels wrap through ±32768.
- The **first triple is the accelerometer**, the second the gyroscope, scaled by the
  full-scale ranges in metadata field 65 (±32 g and ±2000 dps on this unit).

With that reading, accelerometer magnitude holds at **1.008 g** across a whole handheld
clip, and integrating the gyroscope over the same clip gives **1863°** of total rotation
against **1828°** of gravity-vector travel measured independently from the
accelerometer. Swapping the triples, or reading the counts as signed, breaks both
checks. `--inspect` prints the mean magnitude so you can confirm it on your own files.

## Tests

```bash
.venv/bin/pip install -e '.[test]'
.venv/bin/python -m pytest
```

224 tests. Format parsers are tested against byte-exact synthetic trailers, protobuf
messages and real calibration files rather than mocks, so a change in format
understanding fails a test. Tests that need the sample capture or `ffmpeg` skip
themselves when either is absent.

If you run them inside a sourced ROS environment and hit import errors from ROS's own
pytest plugins, clear `PYTHONPATH` for the run: `env PYTHONPATH= .venv/bin/python -m pytest`.

## Layout

| File | Role |
| --- | --- |
| `trailer.py` | Trailer tail, record directory, random access to records |
| `protobuf.py` | Dependency-free protobuf wire decoder (the camera ships no schema) |
| `metadata.py` | Typed view over record `0x0101`, including the wall-clock mapping |
| `calibration.py` | Lens calibration strings → rescaled unified-model intrinsics and their equidistant fit |
| `hdmapping.py` | HDMapping calibration JSON → measured lidar-to-camera pose and intrinsics |
| `sensors.py` | IMU, exposure and preview record decoding |
| `media.py` | ffmpeg pipes for frames and audio, plus JPEG framing |
| `msgdefs.py` | Embedded ROS 2 message definitions |
| `convert.py` | Merges every stream into one time-ordered MCAP |
| `cli.py` | Argument parsing, `--inspect` |

## License

MIT

## Trademarks

This is an independent project, not affiliated with, endorsed by, or sponsored by
Arashi Vision Inc. "Insta360" and the product names above are trademarks of their
respective owners, used here only to identify the file format and hardware this tool
interoperates with. No Insta360 source code, SDK or firmware is included: the trailer
format was derived by observing captures from a camera the author owns.
