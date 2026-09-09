"""Turns an Insta360 ``.insv`` capture into a time-ordered ROS 2 MCAP bag."""

from __future__ import annotations

import contextlib
import heapq
import json
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, List, Optional, Tuple

from mcap.writer import CompressionType
from mcap_ros2.writer import Writer

from . import media, metadata as metadata_mod, msgdefs, sensors, trailer as trailer_mod
from .calibration import DISTORTION_MODEL, LensCalibration, parse_offset_v2

COMPRESSION = {
    "none": CompressionType.NONE,
    "lz4": CompressionType.LZ4,
    "zstd": CompressionType.ZSTD,
}

#: Ordering key for messages that land on the same timestamp, so that a bag is
#: byte-for-byte reproducible from the same input.
_PRIORITY_METADATA = 0
_PRIORITY_CAMERA_INFO = 1
_PRIORITY_IMAGE = 2
_PRIORITY_PREVIEW = 3
_PRIORITY_EXPOSURE = 4
_PRIORITY_IMU = 5
_PRIORITY_AUDIO = 6

#: (topic suffix, frame id suffix) for the two fisheye lenses, in file order.
LENS_NAMES = ("cam_front", "cam_back")

_ZERO_COVARIANCE = [0.0] * 9
_UNKNOWN_ORIENTATION_COVARIANCE = [-1.0] + [0.0] * 8
_IDENTITY_QUATERNION = {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
_NO_ROI = {"x_offset": 0, "y_offset": 0, "height": 0, "width": 0, "do_rectify": False}


@dataclass
class Options:
    input_path: str
    output_path: str
    topic_prefix: str = "/insta360"
    jpeg_quality: int = 3
    scale: Optional[str] = None
    max_frames: Optional[int] = None
    swap_lenses: bool = False
    relative_time: bool = False
    compression: str = "zstd"
    chunk_size: int = 4 << 20
    include_video: bool = True
    include_camera_info: bool = True
    include_imu: bool = True
    include_exposure: bool = True
    include_preview: bool = True
    include_audio: bool = True
    audio_chunk_samples: Optional[int] = None


@dataclass
class Summary:
    output_path: str
    start_ns: int = 0
    end_ns: int = 0
    message_counts: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def total_messages(self) -> int:
        return sum(self.message_counts.values())

    @property
    def duration_s(self) -> float:
        return (self.end_ns - self.start_ns) / 1e9 if self.end_ns > self.start_ns else 0.0


#: An outgoing message: sort keys first so heapq.merge never has to compare payloads.
Emit = Tuple[int, int, int, str, str, dict]


def _stamp(log_time_ns: int) -> Dict[str, int]:
    return {"sec": log_time_ns // 1_000_000_000, "nanosec": log_time_ns % 1_000_000_000}


def _header(log_time_ns: int, frame_id: str) -> Dict[str, object]:
    return {"stamp": _stamp(log_time_ns), "frame_id": frame_id}


def _even(value: float) -> int:
    """Round to a positive even integer; MJPEG's 4:2:0 chroma needs even dimensions."""
    return max(2, int(round(value / 2.0)) * 2)


def resolve_scale(spec: Optional[str], width: int, height: int) -> Tuple[int, int]:
    """Turn a ``--scale`` spec into explicit output dimensions.

    Accepts either a factor (``0.5``) or explicit dimensions (``1440x1440``).  Resolving
    it here rather than letting ffmpeg decide means the intrinsics published in
    CameraInfo always match the frames actually written.
    """
    if not spec:
        return width, height
    text = spec.strip().lower()
    if "x" in text:
        left, _, right = text.partition("x")
        try:
            out_width, out_height = int(left), int(right)
        except ValueError as exc:
            raise ValueError(f"could not parse --scale {spec!r} as WIDTHxHEIGHT") from exc
        if out_width <= 0 or out_height <= 0:
            raise ValueError(f"--scale {spec!r} must be positive")
        return out_width, out_height
    try:
        factor = float(text)
    except ValueError as exc:
        raise ValueError(
            f"could not parse --scale {spec!r} as a factor or WIDTHxHEIGHT"
        ) from exc
    if not 0 < factor <= 1:
        raise ValueError("--scale factor must be greater than 0 and at most 1")
    return _even(width * factor), _even(height * factor)


class _TimeBase:
    """Maps camera device microseconds to the nanosecond stamps written to the bag."""

    def __init__(
        self,
        clock: Optional[metadata_mod.Clock],
        origin_us: int,
        relative: bool,
    ) -> None:
        self._clock = None if relative else clock
        self._origin_us = origin_us

    def __call__(self, device_us: int) -> int:
        if self._clock is not None:
            return self._clock.to_epoch_ns(device_us)
        return (device_us - self._origin_us) * 1000


def convert(options: Options, log: Callable[[str], None] = lambda _: None) -> Summary:
    media.require_tools()
    if options.compression not in COMPRESSION:
        raise ValueError(
            f"unknown compression {options.compression!r}; "
            f"choose one of {', '.join(sorted(COMPRESSION))}"
        )

    summary = Summary(output_path=options.output_path)

    with contextlib.ExitStack() as stack:
        trailer = stack.enter_context(trailer_mod.Trailer(options.input_path))
        for problem in trailer.footer_mismatches():
            summary.warnings.append(f"trailer inconsistency: {problem}")

        meta = metadata_mod.parse(trailer.read(trailer_mod.REC_METADATA))
        probe = media.probe(options.input_path)
        log(
            f"{meta.model or 'unknown camera'} "
            f"(serial {meta.serial or '?'}, firmware {meta.firmware or '?'})"
        )

        if not meta.sensor_ranges_known:
            summary.warnings.append(
                "metadata did not report IMU full-scale ranges; assuming "
                f"+/-{meta.accel_range_g:g} g and +/-{meta.gyro_range_dps:g} dps"
            )

        # -- geometry and frame timing ------------------------------------
        source_width = meta.width or (probe.video[0].width if probe.video else None)
        source_height = meta.height or (probe.video[0].height if probe.video else None)
        if not source_width or not source_height:
            raise ValueError("could not determine the video frame size")
        out_width, out_height = resolve_scale(options.scale, source_width, source_height)
        scale_filter = (
            f"{out_width}:{out_height}"
            if (out_width, out_height) != (source_width, source_height)
            else None
        )

        frame_count = meta.frame_count or 0
        if not frame_count and probe.duration_s and meta.fps:
            frame_count = int(round(probe.duration_s * meta.fps))
        if options.max_frames is not None:
            frame_count = min(frame_count, options.max_frames)
        if options.include_video and frame_count <= 0:
            raise ValueError("could not determine how many video frames the file holds")

        exposures: List[sensors.ExposureSample] = []
        if trailer_mod.REC_EXPOSURE in trailer:
            exposures = sensors.read_exposure(trailer.read(trailer_mod.REC_EXPOSURE))

        frame_us: List[int] = []
        frame_exposure: List[Optional[float]] = []
        if frame_count > 0:
            frame_us, frame_exposure = sensors.frame_timestamps(
                exposures, meta.first_frame_us, frame_count, meta.frame_interval_us
            )
            if all(value is None for value in frame_exposure):
                summary.warnings.append(
                    "no exposure record covering the clip; frame timestamps fall back "
                    "to the nominal frame rate"
                )

        imu_samples: List[sensors.ImuSample] = []
        if options.include_imu and trailer_mod.REC_IMU in trailer:
            imu_samples = sensors.read_imu(
                trailer.read(trailer_mod.REC_IMU),
                meta.accel_range_g,
                meta.gyro_range_dps,
            )
        elif options.include_imu:
            summary.warnings.append("file carries no IMU record")

        # The bag starts at whichever stream begins first; the IMU typically runs for
        # about a second before the first encoded frame.
        candidate_starts = [us for us in (frame_us[:1] or []) if us is not None]
        if imu_samples:
            candidate_starts.append(imu_samples[0].device_us)
        if meta.first_frame_us is not None:
            candidate_starts.append(meta.first_frame_us)
        if not candidate_starts:
            raise ValueError("nothing in this file carries a usable timestamp")
        origin_us = min(candidate_starts)

        clock = meta.clock
        if clock is None and not options.relative_time:
            summary.warnings.append(
                "metadata carries no wall-clock reference; timestamps start at zero"
            )
        time_of = _TimeBase(clock, origin_us, options.relative_time or clock is None)

        # -- lenses --------------------------------------------------------
        lens_order = list(range(len(probe.video)))
        if options.swap_lenses:
            lens_order.reverse()
        lenses = _lens_calibration(meta, out_width, out_height, summary)

        cameras: List[_Camera] = []
        for slot, video_index in enumerate(lens_order[: len(LENS_NAMES)]):
            name = LENS_NAMES[slot]
            cameras.append(
                _Camera(
                    name=name,
                    video_index=video_index,
                    image_topic=f"{options.topic_prefix}/{name}/image/compressed",
                    info_topic=f"{options.topic_prefix}/{name}/camera_info",
                    frame_id=f"insta360_{name}_optical_frame",
                    width=out_width,
                    height=out_height,
                    calibration=lenses.get(video_index),
                )
            )

        # -- writer --------------------------------------------------------
        output = stack.enter_context(open(options.output_path, "wb"))
        writer = Writer(
            output,
            chunk_size=options.chunk_size,
            compression=COMPRESSION[options.compression],
        )
        schemas = {
            name: writer.register_msgdef(name, definition)
            for name, definition in msgdefs.DEFINITIONS.items()
        }

        streams: List[Iterator[Emit]] = []

        # Metadata, first message in the bag.
        streams.append(
            iter([(
                time_of(origin_us),
                _PRIORITY_METADATA,
                0,
                f"{options.topic_prefix}/metadata",
                msgdefs.STRING,
                {"data": json.dumps(_metadata_payload(meta, cameras), indent=2, sort_keys=True)},
            )])
        )

        if options.include_imu and imu_samples:
            streams.append(
                _imu_stream(
                    imu_samples,
                    time_of,
                    topic=f"{options.topic_prefix}/imu",
                    frame_id="insta360_imu",
                )
            )

        if options.include_exposure and frame_us:
            streams.append(
                _exposure_stream(
                    frame_us,
                    frame_exposure,
                    time_of,
                    topic=f"{options.topic_prefix}/exposure_time",
                )
            )

        if options.include_preview and trailer_mod.REC_PREVIEW in trailer:
            streams.append(
                _preview_stream(
                    trailer.read(trailer_mod.REC_PREVIEW),
                    time_of(meta.first_frame_us if meta.first_frame_us is not None else origin_us),
                    topic=f"{options.topic_prefix}/preview/image",
                    frame_id="insta360_preview",
                    summary=summary,
                )
            )

        if options.include_video and cameras and frame_us:
            pipes = [
                stack.enter_context(
                    media.FramePipe(
                        options.input_path,
                        camera.video_index,
                        options.jpeg_quality,
                        scale_filter,
                        limit=frame_count,
                    )
                )
                for camera in cameras
            ]
            streams.append(
                _video_stream(
                    pipes,
                    cameras,
                    frame_us,
                    time_of,
                    include_camera_info=options.include_camera_info,
                    summary=summary,
                    log=log,
                )
            )

        if options.include_audio:
            if probe.audio:
                audio = probe.audio[0]
                sample_rate = audio.sample_rate or 48000
                channels = audio.channels or 2
                chunk = options.audio_chunk_samples or max(
                    1, int(round(sample_rate / (meta.fps or 24.0)))
                )
                duration_s = None
                if options.max_frames is not None and meta.fps:
                    duration_s = frame_count / meta.fps
                pipe = stack.enter_context(
                    media.AudioPipe(
                        options.input_path,
                        audio_index=0,
                        sample_rate=sample_rate,
                        channels=channels,
                        samples_per_chunk=chunk,
                        duration_s=duration_s,
                    )
                )
                streams.append(
                    _audio_stream(
                        pipe,
                        meta.first_frame_us if meta.first_frame_us is not None else origin_us,
                        time_of,
                        topic=f"{options.topic_prefix}/audio",
                    )
                )
            else:
                summary.warnings.append("file carries no audio stream")

        # -- merge and write ------------------------------------------------
        first_ns: Optional[int] = None
        last_ns = 0
        sequences: Dict[str, int] = {}
        for log_time_ns, _priority, _seq, topic, type_name, message in heapq.merge(*streams):
            sequence = sequences.get(topic, 0)
            sequences[topic] = sequence + 1
            writer.write_message(
                topic,
                schemas[type_name],
                message,
                log_time=log_time_ns,
                publish_time=log_time_ns,
                sequence=sequence,
            )
            summary.message_counts[topic] = summary.message_counts.get(topic, 0) + 1
            if first_ns is None:
                first_ns = log_time_ns
            last_ns = log_time_ns

        writer.finish()
        summary.start_ns = first_ns or 0
        summary.end_ns = last_ns

    return summary


# -- helpers ----------------------------------------------------------------


@dataclass
class _Camera:
    name: str
    video_index: int
    image_topic: str
    info_topic: str
    frame_id: str
    width: int
    height: int
    calibration: Optional[LensCalibration]


def _lens_calibration(
    meta: metadata_mod.Metadata,
    width: int,
    height: int,
    summary: Summary,
) -> Dict[int, LensCalibration]:
    text = meta.calibration_strings.get("offset_v2")
    if not text:
        summary.warnings.append("no lens calibration in metadata; CameraInfo omitted")
        return {}
    lenses = parse_offset_v2(text, width, height)
    if not lenses:
        summary.warnings.append(
            "lens calibration string had an unexpected layout; CameraInfo omitted"
        )
        return {}
    return {lens.index: lens for lens in lenses}


def _metadata_payload(
    meta: metadata_mod.Metadata, cameras: List[_Camera]
) -> Dict[str, object]:
    payload = meta.as_dict()
    payload["imu"] = {
        "frame_id": "insta360_imu",
        "units": "linear_acceleration in m/s^2, angular_velocity in rad/s",
        "axes": "raw sensor axes, NOT rotated into REP 103",
    }
    payload["lenses"] = [
        {
            "topic": camera.image_topic,
            "frame_id": camera.frame_id,
            "video_stream_index": camera.video_index,
            "width": camera.width,
            "height": camera.height,
            "intrinsics": None
            if camera.calibration is None
            else {
                "fx": camera.calibration.fx,
                "fy": camera.calibration.fy,
                "cx": camera.calibration.cx,
                "cy": camera.calibration.cy,
                "distortion_model": DISTORTION_MODEL,
                "distortion": camera.calibration.distortion,
                "rotation_deg": list(camera.calibration.rotation_deg),
                "translation": list(camera.calibration.translation),
            },
        }
        for camera in cameras
    ]
    return payload


def _imu_stream(
    samples: List[sensors.ImuSample],
    time_of: _TimeBase,
    topic: str,
    frame_id: str,
) -> Iterator[Emit]:
    for index, sample in enumerate(samples):
        log_time = time_of(sample.device_us)
        yield (
            log_time,
            _PRIORITY_IMU,
            index,
            topic,
            msgdefs.IMU,
            {
                "header": _header(log_time, frame_id),
                "orientation": _IDENTITY_QUATERNION,
                "orientation_covariance": _UNKNOWN_ORIENTATION_COVARIANCE,
                "angular_velocity": {
                    "x": sample.gyro[0], "y": sample.gyro[1], "z": sample.gyro[2],
                },
                "angular_velocity_covariance": _ZERO_COVARIANCE,
                "linear_acceleration": {
                    "x": sample.accel[0], "y": sample.accel[1], "z": sample.accel[2],
                },
                "linear_acceleration_covariance": _ZERO_COVARIANCE,
            },
        )


def _exposure_stream(
    frame_us: List[int],
    frame_exposure: List[Optional[float]],
    time_of: _TimeBase,
    topic: str,
) -> Iterator[Emit]:
    for index, (device_us, exposure) in enumerate(zip(frame_us, frame_exposure)):
        if exposure is None:
            continue
        log_time = time_of(device_us)
        yield (log_time, _PRIORITY_EXPOSURE, index, topic, msgdefs.FLOAT64, {"data": exposure})


def _preview_stream(
    raw: bytes,
    log_time: int,
    topic: str,
    frame_id: str,
    summary: Summary,
) -> Iterator[Emit]:
    try:
        preview = sensors.read_preview(raw)
        rgb = media.nv12_to_rgb8(preview.nv12, preview.width, preview.height)
    except (ValueError, media.FFmpegError) as exc:
        summary.warnings.append(f"could not decode the embedded preview image: {exc}")
        return
    yield (
        log_time,
        _PRIORITY_PREVIEW,
        0,
        topic,
        msgdefs.IMAGE,
        {
            "header": _header(log_time, frame_id),
            "height": preview.height,
            "width": preview.width,
            "encoding": "rgb8",
            "is_bigendian": 0,
            "step": preview.width * 3,
            "data": rgb,
        },
    )


def _video_stream(
    pipes: List[media.FramePipe],
    cameras: List[_Camera],
    frame_us: List[int],
    time_of: _TimeBase,
    include_camera_info: bool,
    summary: Summary,
    log: Callable[[str], None],
) -> Iterator[Emit]:
    iterators = [iter(pipe) for pipe in pipes]
    total = len(frame_us)
    step = max(1, total // 10)
    written = 0
    for index, device_us in enumerate(frame_us):
        frames: List[Optional[bytes]] = [next(it, None) for it in iterators]
        if any(frame is None for frame in frames):
            missing = [camera.name for camera, frame in zip(cameras, frames) if frame is None]
            summary.warnings.append(
                f"video ended after {index} of {total} frames "
                f"(no more data from {', '.join(missing)})"
            )
            break
        log_time = time_of(device_us)
        for camera, frame in zip(cameras, frames):
            if include_camera_info and camera.calibration is not None:
                yield (
                    log_time,
                    _PRIORITY_CAMERA_INFO,
                    index,
                    camera.info_topic,
                    msgdefs.CAMERA_INFO,
                    {
                        "header": _header(log_time, camera.frame_id),
                        "height": camera.height,
                        "width": camera.width,
                        "distortion_model": DISTORTION_MODEL,
                        "d": camera.calibration.distortion,
                        "k": camera.calibration.k,
                        "r": camera.calibration.r,
                        "p": camera.calibration.p,
                        "binning_x": 0,
                        "binning_y": 0,
                        "roi": _NO_ROI,
                    },
                )
            yield (
                log_time,
                _PRIORITY_IMAGE,
                index,
                camera.image_topic,
                msgdefs.COMPRESSED_IMAGE,
                {
                    "header": _header(log_time, camera.frame_id),
                    "format": "jpeg",
                    "data": frame,
                },
            )
        written = index + 1
        if written % step == 0 or written == total:
            log(f"  video {written}/{total} frames ({100.0 * written / total:.0f}%)")


def _audio_stream(
    pipe: media.AudioPipe,
    first_frame_us: int,
    time_of: _TimeBase,
    topic: str,
) -> Iterator[Emit]:
    for index, (sample_index, block) in enumerate(pipe):
        device_us = first_frame_us + round(sample_index * 1_000_000 / pipe.sample_rate)
        log_time = time_of(device_us)
        yield (log_time, _PRIORITY_AUDIO, index, topic, msgdefs.AUDIO_DATA, {"data": block})
