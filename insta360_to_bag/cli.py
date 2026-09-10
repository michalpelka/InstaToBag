"""Command line interface for insta360-to-bag."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from typing import List, Optional

from . import media, metadata as metadata_mod, sensors, trailer as trailer_mod
from .calibration import summarise
from .convert import COMPRESSION, DEFAULT_CAMERA_RPY, Options, convert, resolve_scale

DESCRIPTION = """\
Convert an Insta360 .insv capture into a ROS 2 MCAP bag.

Writes both fisheye tracks as sensor_msgs/CompressedImage, the 1 kHz IMU as
sensor_msgs/Imu, per-lens intrinsics as sensor_msgs/CameraInfo, per-frame exposure,
the embedded equirectangular preview, and the audio track -- all on one time base
recovered from the camera's own clock.
"""


def _human_bytes(size: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} GiB"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="insta360-to-bag",
        description=DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="path to the .insv (or .insp) file")
    parser.add_argument(
        "-o", "--output",
        help="output .mcap path (default: the input path with an .mcap suffix)",
    )
    parser.add_argument(
        "--inspect", action="store_true",
        help="print what the file contains and exit without writing a bag",
    )

    group = parser.add_argument_group("video")
    group.add_argument(
        "--jpeg-quality", type=int, default=3, metavar="N",
        help="ffmpeg MJPEG quality, 2 (best) to 31 (worst); default 3",
    )
    group.add_argument(
        "--scale", metavar="SPEC",
        help="downscale frames, either as a factor (0.5) or explicit size (1440x1440)",
    )
    group.add_argument(
        "--max-frames", type=int, metavar="N",
        help="stop after N video frames; useful for a quick look at a long clip",
    )
    group.add_argument(
        "--swap-lenses", action="store_true",
        help="map the second video track to cam_front instead of the first",
    )

    group = parser.add_argument_group("contents")
    for name, help_text in (
        ("video", "the two fisheye image topics"),
        ("camera-info", "per-lens CameraInfo"),
        ("imu", "the 1 kHz IMU"),
        ("exposure", "per-frame exposure time"),
        ("preview", "the embedded equirectangular preview image"),
        ("audio", "the audio track"),
        ("tf", "the static lidar-to-camera transforms on /tf_static"),
    ):
        group.add_argument(
            f"--no-{name}", dest=name.replace("-", "_"), action="store_false",
            help=f"leave out {help_text}",
        )

    group = parser.add_argument_group("extrinsics")
    group.add_argument(
        "--lidar-frame", default="lidar", metavar="FRAME",
        help="parent frame of the static camera transforms; default lidar",
    )
    group.add_argument(
        "--camera-xyz", type=float, nargs=3, default=[0.0, 0.0, 0.0],
        metavar=("X", "Y", "Z"),
        help="camera centre in the lidar frame, in metres; default 0 0 0",
    )
    group.add_argument(
        "--camera-rpy", type=float, nargs=3, default=list(DEFAULT_CAMERA_RPY),
        metavar=("ROLL", "PITCH", "YAW"),
        help="camera body orientation in the lidar frame, as URDF fixed-axis roll pitch "
             "yaw in degrees; body x is the front lens's view, z the lens end. Default "
             "90 0 90: on its side, lens end toward +x, front lens looking left. "
             "Upright with the front lens looking left is 0 0 90",
    )

    group = parser.add_argument_group("output format")
    group.add_argument(
        "--topic-prefix", default="/insta360",
        help="prefix for every topic name; default /insta360",
    )
    group.add_argument(
        "--compression", default="zstd", choices=sorted(COMPRESSION),
        help="MCAP chunk compression; default zstd",
    )
    group.add_argument(
        "--chunk-size", type=int, default=4 << 20, metavar="BYTES",
        help="MCAP chunk size in bytes; default 4194304",
    )
    group.add_argument(
        "--relative-time", action="store_true",
        help="start timestamps at zero instead of the camera's wall clock",
    )
    group.add_argument(
        "--audio-chunk-samples", type=int, metavar="N",
        help="samples per audio message; default is one video frame's worth",
    )
    parser.add_argument(
        "-f", "--force", action="store_true",
        help="overwrite the output file if it already exists",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="only report warnings and errors",
    )
    return parser


def _inspect(path: str) -> int:
    with trailer_mod.Trailer(path) as trailer:
        meta = metadata_mod.parse(trailer.read(trailer_mod.REC_METADATA))
        probe = media.probe(path)

        print(f"file          {path}")
        print(f"size          {_human_bytes(trailer.file_size)}")
        print(f"trailer       version {trailer.version}, "
              f"{_human_bytes(trailer.file_size - trailer.base_offset)} "
              f"at offset {trailer.base_offset}")
        print()
        print(f"camera        {meta.model or '?'}  serial {meta.serial or '?'}")
        print(f"firmware      {meta.firmware or '?'}")
        print(f"captured      {meta.capture_datetime or '?'} (camera local time)")
        if meta.start_epoch_ms:
            started = dt.datetime.fromtimestamp(meta.start_epoch_ms / 1000, dt.timezone.utc)
            print(f"first frame   {started.isoformat()} (utc)")
        print(f"profile       {meta.profile or '?'}")
        print(f"source        {meta.source_path or '?'}")
        print()
        print(f"frames        {meta.frame_count or '?'} at {meta.fps or '?'} fps, "
              f"{meta.width or '?'}x{meta.height or '?'} per lens")
        print(f"duration      {meta.duration_s if meta.duration_s is not None else '?'} s")
        print(f"imu ranges    +/-{meta.accel_range_g:g} g, +/-{meta.gyro_range_dps:g} dps"
              + ("" if meta.sensor_ranges_known else "  (assumed; not in metadata)"))
        print()

        print("container streams")
        for stream in probe.video:
            print(f"  video {stream.index}  {stream.codec} "
                  f"{stream.width}x{stream.height} @ {stream.frame_rate or '?':.6g} fps")
        for stream in probe.audio:
            print(f"  audio {stream.index}  {stream.codec} "
                  f"{stream.sample_rate} Hz, {stream.channels} ch")
        print()

        print("trailer records")
        for record in trailer:
            print(f"  0x{record.id:04x}  {_human_bytes(record.length):>10}  {record.name}")
        print()

        if trailer_mod.REC_IMU in trailer:
            samples = sensors.read_imu(
                trailer.read(trailer_mod.REC_IMU), meta.accel_range_g, meta.gyro_range_dps
            )
            if samples:
                span = (samples[-1].device_us - samples[0].device_us) / 1e6
                rate = (len(samples) - 1) / span if span > 0 else 0.0
                magnitudes = [
                    (s.accel[0] ** 2 + s.accel[1] ** 2 + s.accel[2] ** 2) ** 0.5
                    for s in samples
                ]
                mean_g = sum(magnitudes) / len(magnitudes) / sensors.STANDARD_GRAVITY
                print(f"imu           {len(samples)} samples over {span:.3f} s "
                      f"({rate:.1f} Hz)")
                print(f"              mean |accel| = {mean_g:.4f} g "
                      "(should sit near 1.0 for a handheld clip)")

        if trailer_mod.REC_EXPOSURE in trailer:
            exposures = sensors.read_exposure(trailer.read(trailer_mod.REC_EXPOSURE))
            if exposures:
                values = [s.exposure_s * 1000 for s in exposures]
                print(f"exposure      {len(exposures)} entries, "
                      f"{min(values):.3f} to {max(values):.3f} ms")

        if trailer_mod.REC_PREVIEW in trailer:
            try:
                preview = sensors.read_preview(trailer.read(trailer_mod.REC_PREVIEW))
                print(f"preview       {preview.width}x{preview.height} NV12 "
                      "equirectangular")
            except ValueError as exc:
                print(f"preview       unreadable: {exc}")

        if meta.lenses:
            print()
            print("calibration (rescaled to the stored frame size)")
            for lens in meta.lenses:
                print(f"  {summarise(lens)}")
                print(f"    unified k1 k2 k3 p1 p2 {lens.distortion}")
                print(f"    published as equidistant fx={lens.equidistant_fx:.2f} "
                      f"fy={lens.equidistant_fy:.2f} "
                      f"k1..k4 {[round(c, 8) for c in lens.equidistant]}")
                print(f"    fit within {lens.fit_error_px:.2f} px out to "
                      f"{lens.fit_max_angle_deg:.1f} deg off-axis")

        problems = trailer.footer_mismatches()
        if problems:
            print()
            print("warnings")
            for problem in problems:
                print(f"  {problem}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not os.path.exists(args.input):
        parser.error(f"input file not found: {args.input}")

    try:
        media.require_tools()
    except media.FFmpegError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.inspect:
        try:
            return _inspect(args.input)
        except (trailer_mod.TrailerError, media.FFmpegError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    output = args.output or os.path.splitext(args.input)[0] + ".mcap"
    if os.path.exists(output) and not args.force:
        print(
            f"error: {output} already exists; pass --force to overwrite",
            file=sys.stderr,
        )
        return 2

    if args.scale:
        try:
            resolve_scale(args.scale, 100, 100)
        except ValueError as exc:
            parser.error(str(exc))
    if not args.lidar_frame or args.lidar_frame.startswith("/"):
        parser.error("--lidar-frame must be a non-empty frame id without a leading '/'")

    options = Options(
        input_path=args.input,
        output_path=output,
        topic_prefix=args.topic_prefix.rstrip("/"),
        jpeg_quality=args.jpeg_quality,
        scale=args.scale,
        max_frames=args.max_frames,
        swap_lenses=args.swap_lenses,
        relative_time=args.relative_time,
        compression=args.compression,
        chunk_size=args.chunk_size,
        include_video=args.video,
        include_camera_info=args.camera_info,
        include_imu=args.imu,
        include_exposure=args.exposure,
        include_preview=args.preview,
        include_audio=args.audio,
        include_tf=args.tf,
        audio_chunk_samples=args.audio_chunk_samples,
        lidar_frame=args.lidar_frame,
        camera_xyz=tuple(args.camera_xyz),
        camera_rpy=tuple(args.camera_rpy),
    )

    log = (lambda _: None) if args.quiet else (lambda message: print(message, flush=True))

    try:
        summary = convert(options, log=log)
    except (trailer_mod.TrailerError, media.FFmpegError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted; the partial bag is incomplete", file=sys.stderr)
        return 130

    if not args.quiet:
        print()
        print(f"wrote {summary.output_path} "
              f"({_human_bytes(os.path.getsize(summary.output_path))}, "
              f"{summary.total_messages} messages, {summary.duration_s:.3f} s)")
        for topic in sorted(summary.message_counts):
            print(f"  {summary.message_counts[topic]:>7}  {topic}")

    for warning in summary.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    return 0
