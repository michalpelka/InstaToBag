"""ffmpeg-driven extraction of the audio/video essence from an ``.insv`` file.

The container is a plain MP4, so ffmpeg reads it directly; only the trailer needs
bespoke parsing.  Everything here streams through pipes so that a long clip never has
to be staged on disk or held in memory.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Iterator, List, Optional

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"

_READ_CHUNK = 4 << 20


class FFmpegError(RuntimeError):
    pass


@dataclass(frozen=True)
class StreamInfo:
    index: int
    codec: Optional[str]
    width: Optional[int] = None
    height: Optional[int] = None
    frame_rate: Optional[float] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None


@dataclass(frozen=True)
class Probe:
    duration_s: Optional[float]
    video: List[StreamInfo]
    audio: List[StreamInfo]


def require_tools() -> None:
    missing = [tool for tool in (FFMPEG, FFPROBE) if shutil.which(tool) is None]
    if missing:
        raise FFmpegError(
            f"{' and '.join(missing)} not found on PATH; install ffmpeg "
            "(e.g. 'sudo apt install ffmpeg')"
        )


def _parse_rate(text: Optional[str]) -> Optional[float]:
    if not text or "/" not in text:
        return None
    numerator, _, denominator = text.partition("/")
    try:
        den = float(denominator)
        return float(numerator) / den if den else None
    except ValueError:
        return None


def probe(path: str) -> Probe:
    """Enumerate the streams in the container via ``ffprobe``."""
    command = [
        FFPROBE, "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", path,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise FFmpegError(f"ffprobe failed on {path}: {result.stderr.strip()}")
    try:
        parsed = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise FFmpegError(f"could not parse ffprobe output: {exc}") from exc

    video: List[StreamInfo] = []
    audio: List[StreamInfo] = []
    for stream in parsed.get("streams", []):
        kind = stream.get("codec_type")
        if kind == "video":
            video.append(
                StreamInfo(
                    index=int(stream["index"]),
                    codec=stream.get("codec_name"),
                    width=stream.get("width"),
                    height=stream.get("height"),
                    frame_rate=_parse_rate(stream.get("avg_frame_rate")),
                )
            )
        elif kind == "audio":
            audio.append(
                StreamInfo(
                    index=int(stream["index"]),
                    codec=stream.get("codec_name"),
                    sample_rate=int(stream["sample_rate"]) if stream.get("sample_rate") else None,
                    channels=stream.get("channels"),
                )
            )
    duration = parsed.get("format", {}).get("duration")
    try:
        duration_s = float(duration) if duration is not None else None
    except (TypeError, ValueError):
        duration_s = None
    return Probe(duration_s=duration_s, video=video, audio=audio)


# -- JPEG framing -----------------------------------------------------------


def _find_jpeg_end(buf: bytearray, start: int) -> int:
    """Return the index just past the EOI of the JPEG at ``start``, or -1 if truncated.

    Walks the marker structure rather than searching for a bare ``FF D9``: inside
    entropy-coded scan data a literal 0xFF is byte-stuffed as ``FF 00``, so a naive
    search happens to work for ffmpeg's output, but restart markers and multi-scan
    images make that a coincidence rather than a guarantee.
    """
    size = len(buf)
    pos = start
    if size - pos < 2 or buf[pos] != 0xFF or buf[pos + 1] != 0xD8:
        raise FFmpegError("ffmpeg output is not positioned at a JPEG SOI marker")
    pos += 2
    while True:
        while pos < size and buf[pos] != 0xFF:
            pos += 1
        while pos < size and buf[pos] == 0xFF:  # 0xFF fill bytes before a marker
            pos += 1
        if pos >= size:
            return -1
        marker = buf[pos]
        pos += 1
        if marker == 0xD9:  # EOI
            return pos
        if marker == 0x01 or 0xD0 <= marker <= 0xD7:  # TEM, RSTn: no payload
            continue
        if size - pos < 2:
            return -1
        segment_length = (buf[pos] << 8) | buf[pos + 1]
        if segment_length < 2:
            raise FFmpegError("invalid JPEG segment length in ffmpeg output")
        pos += segment_length
        if marker != 0xDA:  # not SOS, so the next marker follows immediately
            continue
        # Scan past entropy-coded data to the next real marker.
        while True:
            found = buf.find(b"\xff", pos)
            if found < 0 or found + 1 >= size:
                return -1
            following = buf[found + 1]
            if following == 0x00 or 0xD0 <= following <= 0xD7 or following == 0xFF:
                pos = found + 1 if following == 0xFF else found + 2
                continue
            pos = found
            break


class FramePipe:
    """Streams one video track out of the file as a sequence of JPEG buffers."""

    def __init__(
        self,
        path: str,
        video_index: int,
        quality: int,
        scale_filter: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> None:
        self.path = path
        command = [FFMPEG, "-nostdin", "-v", "error", "-i", path, "-map", f"0:v:{video_index}"]
        if limit is not None:
            command += ["-frames:v", str(limit)]
        if scale_filter:
            command += ["-vf", f"scale={scale_filter}"]
        command += ["-c:v", "mjpeg", "-q:v", str(quality), "-f", "image2pipe", "-"]
        self._command = command
        self._stderr = tempfile.TemporaryFile()
        self._process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=self._stderr
        )
        self._exhausted = False

    def __enter__(self) -> "FramePipe":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __iter__(self) -> Iterator[bytes]:
        assert self._process.stdout is not None
        buf = bytearray()
        while True:
            end = -1
            if buf:
                end = _find_jpeg_end(buf, 0)
            if end < 0:
                chunk = self._process.stdout.read(_READ_CHUNK)
                if not chunk:
                    break
                buf += chunk
                continue
            yield bytes(buf[:end])
            del buf[:end]
        if buf.strip(b"\x00"):
            raise FFmpegError(
                f"{len(buf)} trailing bytes from ffmpeg did not form a complete JPEG"
            )
        self._exhausted = True

    def _stderr_tail(self, limit: int = 2000) -> str:
        try:
            self._stderr.seek(0)
            return self._stderr.read().decode("utf-8", "replace")[-limit:].strip()
        except OSError:
            return ""

    def close(self) -> None:
        if self._process.poll() is None:
            if not self._exhausted:
                self._process.kill()
            self._process.wait()
        if self._process.stdout is not None:
            self._process.stdout.close()
        code = self._process.returncode
        tail = self._stderr_tail()
        self._stderr.close()
        # A non-zero exit only matters if we let ffmpeg run to completion; when the
        # caller stops early we kill it deliberately.
        if self._exhausted and code not in (0, None):
            raise FFmpegError(
                f"ffmpeg exited with status {code} while decoding {self.path}"
                + (f":\n{tail}" if tail else "")
            )


# -- audio ------------------------------------------------------------------


class AudioPipe:
    """Streams the audio track out as fixed-size blocks of interleaved PCM s16le."""

    def __init__(
        self,
        path: str,
        audio_index: int,
        sample_rate: int,
        channels: int,
        samples_per_chunk: int,
        duration_s: Optional[float] = None,
    ) -> None:
        self.path = path
        self.sample_rate = sample_rate
        self.channels = channels
        self.samples_per_chunk = samples_per_chunk
        self.bytes_per_sample = 2 * channels
        command = [
            FFMPEG, "-nostdin", "-v", "error", "-i", path,
            "-map", f"0:a:{audio_index}", "-vn",
        ]
        if duration_s is not None:
            command += ["-t", f"{duration_s:.6f}"]
        command += [
            "-f", "s16le", "-acodec", "pcm_s16le",
            "-ar", str(sample_rate), "-ac", str(channels), "-",
        ]
        self._stderr = tempfile.TemporaryFile()
        self._process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=self._stderr
        )
        self._exhausted = False

    def __enter__(self) -> "AudioPipe":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __iter__(self) -> Iterator[tuple[int, bytes]]:
        """Yield ``(first_sample_index, pcm_bytes)`` for each block."""
        assert self._process.stdout is not None
        block_bytes = self.samples_per_chunk * self.bytes_per_sample
        sample_index = 0
        while True:
            block = self._read_exactly(block_bytes)
            if not block:
                break
            # Never split a frame across blocks; drop a trailing partial frame.
            usable = len(block) - (len(block) % self.bytes_per_sample)
            if usable == 0:
                break
            yield sample_index, block[:usable]
            sample_index += usable // self.bytes_per_sample
            if usable < block_bytes:
                break
        self._exhausted = True

    def _read_exactly(self, size: int) -> bytes:
        assert self._process.stdout is not None
        parts: List[bytes] = []
        remaining = size
        while remaining > 0:
            chunk = self._process.stdout.read(remaining)
            if not chunk:
                break
            parts.append(chunk)
            remaining -= len(chunk)
        return b"".join(parts)

    def close(self) -> None:
        if self._process.poll() is None:
            if not self._exhausted:
                self._process.kill()
            self._process.wait()
        if self._process.stdout is not None:
            self._process.stdout.close()
        code = self._process.returncode
        try:
            self._stderr.seek(0)
            tail = self._stderr.read().decode("utf-8", "replace")[-2000:].strip()
        except OSError:
            tail = ""
        self._stderr.close()
        if self._exhausted and code not in (0, None):
            raise FFmpegError(
                f"ffmpeg exited with status {code} while decoding audio from {self.path}"
                + (f":\n{tail}" if tail else "")
            )


# -- pixel format conversion ------------------------------------------------


def nv12_to_rgb8(nv12: bytes, width: int, height: int) -> bytes:
    """Convert an NV12 buffer to packed rgb8 using ffmpeg's colour conversion.

    Done through ffmpeg rather than by hand so that the YUV matrix and range match
    what every other tool shows for this file.
    """
    command = [
        FFMPEG, "-nostdin", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "nv12", "-s", f"{width}x{height}", "-i", "-",
        "-frames:v", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "-",
    ]
    result = subprocess.run(command, input=nv12, capture_output=True)
    expected = width * height * 3
    if result.returncode != 0 or len(result.stdout) != expected:
        raise FFmpegError(
            "NV12 to RGB conversion failed: "
            + (result.stderr.decode("utf-8", "replace").strip() or
               f"got {len(result.stdout)} bytes, expected {expected}")
        )
    return result.stdout
