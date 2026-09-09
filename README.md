# insta360-to-bag

Convert an Insta360 `.insv` capture into a ROS 2 MCAP bag: both fisheye tracks, the
1 kHz IMU, per-lens intrinsics, per-frame exposure, the embedded preview image and the
audio track, all on a single time base recovered from the camera's own clock.

Developed and verified against an **Insta360 X5** (firmware `v1.10.11_build1`, 5.7K
dual-fisheye, 2880×2880 per lens at 24 fps). The trailer format is shared across the
X-series, so ONE X2/X3/X4 files should work; the parsers degrade to "field not present"
rather than to wrong values when a firmware revision differs.

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
```

`ffmpeg` and `ffprobe` must be on `PATH` (`sudo apt install ffmpeg`). No ROS
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
```

Run `insta360-to-bag --help` for the full set. The useful ones:

| Flag | Effect |
| --- | --- |
| `--jpeg-quality N` | ffmpeg MJPEG quality, 2 (best) to 31. Default 3. |
| `--scale SPEC` | `0.5` or `1440x1440`. Intrinsics are rescaled to match. |
| `--max-frames N` | Stop after N frames. |
| `--swap-lenses` | Map the second video track to `cam_front`. |
| `--no-{video,camera-info,imu,exposure,preview,audio}` | Leave a stream out. |
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

Messages are written in strict time order, and each message's `header.stamp` equals its
MCAP log time.

A 15.6 s 5.7K clip becomes roughly 350 MB at the default JPEG quality; conversion is
bounded by HEVC decode of two 2880×2880 tracks.

### Frames

`insta360_cam_front_optical_frame`, `insta360_cam_back_optical_frame`, `insta360_imu`.
No transform between them is published — see the caveats below.

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
  calibrated IMU-to-camera transform, so no `tf` is emitted. The per-lens rotation and
  translation values are passed through on the metadata topic.
- **The distortion model is Insta360's, not OpenCV's.** `CameraInfo.distortion_model`
  is set to `insta360_fisheye_v2` and `d` holds the camera's own five coefficients.
  They are *not* `plumb_bob` or `equidistant` and must not be fed to
  `cv::undistort`/`cv::fisheye` as if they were. `K` and `P` are ordinary pinhole
  intrinsics and are safe to use. The raw calibration strings are republished verbatim
  on `/insta360/metadata`.
- **`cam_front` / `cam_back` is naming, not a determination.** They follow the order of
  the video tracks in the file, which is stable but has not been confirmed against
  which lens physically faces the screen. Use `--swap-lenses` if you need them the
  other way round.
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

138 tests. Format parsers are tested against byte-exact synthetic trailers and
protobuf messages rather than mocks, so a change in format understanding fails a test.
Tests that need the sample capture or `ffmpeg` skip themselves when either is absent.

If you run them inside a sourced ROS environment and hit import errors from ROS's own
pytest plugins, clear `PYTHONPATH` for the run: `env PYTHONPATH= .venv/bin/python -m pytest`.

## Layout

| File | Role |
| --- | --- |
| `trailer.py` | Trailer tail, record directory, random access to records |
| `protobuf.py` | Dependency-free protobuf wire decoder (the camera ships no schema) |
| `metadata.py` | Typed view over record `0x0101`, including the wall-clock mapping |
| `calibration.py` | Lens calibration strings → rescaled intrinsics |
| `sensors.py` | IMU, exposure and preview record decoding |
| `media.py` | ffmpeg pipes for frames and audio, plus JPEG framing |
| `msgdefs.py` | Embedded ROS 2 message definitions |
| `convert.py` | Merges every stream into one time-ordered MCAP |
| `cli.py` | Argument parsing, `--inspect` |

## License

MIT
