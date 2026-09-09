"""Reader for the Insta360 metadata trailer appended to ``.insv`` / ``.insp`` files.

An ``.insv`` file is an ordinary MP4 (two fisheye HEVC tracks plus AAC audio) with a
proprietary blob glued onto the end.  The layout, walking backwards from EOF, is::

    ...MP4 boxes... | record | footer | record | footer | ... | directory | footer | tail

    tail (72 bytes)   32 bytes reserved (zero)
                      uint32  trailer size, in bytes, counted back from EOF
                      uint32  trailer format version (3 on current firmware)
                      char[32] b"8db42d694ccc418790edff439fe026bf"

    footer (6 bytes)  uint16 be  record id
                      uint32 le  record payload length

    directory         10 bytes reserved, then 10-byte entries of
                      uint16 le record id, uint32 le length, uint32 le offset
                      where offset is relative to the start of the trailer

Note the record id is big-endian in the per-record footer but little-endian in the
directory -- verified against real captures.  The directory is authoritative here and
the footers are used only as a consistency check.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional

MAGIC = b"8db42d694ccc418790edff439fe026bf"
TAIL_LEN = 32 + 4 + 4 + len(MAGIC)  # 72
FOOTER_LEN = 6
DIRECTORY_HEADER_LEN = 10
DIRECTORY_ENTRY_LEN = 10

# Record ids seen on Insta360 ONE X2 / X3 / X4 / X5 firmware.
REC_DIRECTORY = 0x0000
REC_PREVIEW = 0x0002  # NV12 equirectangular thumbnail
REC_IMU = 0x0003  # 1 kHz accelerometer + gyroscope
REC_EXPOSURE = 0x0004  # per-frame exposure time
REC_ISP_STATS_A = 0x0009
REC_SENSOR_MAP = 0x000A
REC_ISP_STATS_B = 0x000B
REC_PREVIEW_SEQUENCE = 0x0016
REC_LUT = 0x001C
REC_FRAME_MARKS = 0x001D
REC_METADATA = 0x0101  # protobuf; see metadata.py

RECORD_NAMES = {
    REC_DIRECTORY: "directory",
    REC_PREVIEW: "preview image (NV12)",
    REC_IMU: "imu (accel + gyro)",
    REC_EXPOSURE: "exposure times",
    REC_ISP_STATS_A: "isp stats a",
    REC_SENSOR_MAP: "sensor map",
    REC_ISP_STATS_B: "isp stats b",
    REC_PREVIEW_SEQUENCE: "preview sequence",
    REC_LUT: "lookup table",
    REC_FRAME_MARKS: "frame marks",
    REC_METADATA: "metadata (protobuf)",
}


class TrailerError(ValueError):
    """Raised when a file carries no usable Insta360 trailer."""


@dataclass(frozen=True)
class Record:
    """One trailer record: where it lives in the file and how long it is."""

    id: int
    offset: int  # absolute file offset of the payload
    length: int

    @property
    def name(self) -> str:
        return RECORD_NAMES.get(self.id, f"unknown 0x{self.id:04x}")


class Trailer:
    """Random-access reader over the trailer records of a single file."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.file_size = os.path.getsize(path)
        self._fh = open(path, "rb")
        try:
            self.version, self.base_offset = self._read_tail()
            self.records: Dict[int, Record] = self._read_directory()
        except Exception:
            self._fh.close()
            raise

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "Trailer":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- parsing -----------------------------------------------------------

    def _read_at(self, offset: int, length: int) -> bytes:
        self._fh.seek(offset)
        data = self._fh.read(length)
        if len(data) != length:
            raise TrailerError(f"short read of {length} bytes at offset {offset}")
        return data

    def _read_tail(self) -> tuple[int, int]:
        if self.file_size < TAIL_LEN:
            raise TrailerError("file is too small to contain an Insta360 trailer")
        tail = self._read_at(self.file_size - TAIL_LEN, TAIL_LEN)
        if not tail.endswith(MAGIC):
            raise TrailerError(
                "Insta360 trailer magic not found at end of file -- "
                "is this an original .insv/.insp straight off the camera?"
            )
        trailer_size, version = struct.unpack_from("<II", tail, 32)
        if not 0 < trailer_size <= self.file_size:
            raise TrailerError(f"implausible trailer size {trailer_size}")
        return version, self.file_size - trailer_size

    def _read_directory(self) -> Dict[int, Record]:
        # The directory is the last record, immediately before the 72-byte tail.
        footer_at = self.file_size - TAIL_LEN - FOOTER_LEN
        dir_id, dir_len = self._unpack_footer(footer_at)
        if dir_id != REC_DIRECTORY:
            raise TrailerError(f"expected directory record, found id 0x{dir_id:04x}")
        raw = self._read_at(footer_at - dir_len, dir_len)

        records: Dict[int, Record] = {}
        body = raw[DIRECTORY_HEADER_LEN:]
        for pos in range(0, len(body) - DIRECTORY_ENTRY_LEN + 1, DIRECTORY_ENTRY_LEN):
            rec_id, length, offset = struct.unpack_from("<HII", body, pos)
            if rec_id == 0 and length == 0:
                continue  # unused slot; the table is fixed-size and zero padded
            absolute = self.base_offset + offset
            if absolute < 0 or absolute + length > self.file_size:
                raise TrailerError(
                    f"directory entry 0x{rec_id:04x} points outside the file"
                )
            records[rec_id] = Record(id=rec_id, offset=absolute, length=length)
        if not records:
            raise TrailerError("trailer directory is empty")
        return records

    def _unpack_footer(self, offset: int) -> tuple[int, int]:
        raw = self._read_at(offset, FOOTER_LEN)
        return struct.unpack(">H", raw[:2])[0], struct.unpack("<I", raw[2:])[0]

    # -- access ------------------------------------------------------------

    def __contains__(self, record_id: int) -> bool:
        return record_id in self.records

    def __iter__(self) -> Iterator[Record]:
        return iter(sorted(self.records.values(), key=lambda r: r.offset))

    def get(self, record_id: int) -> Optional[Record]:
        return self.records.get(record_id)

    def read(self, record_id: int) -> bytes:
        record = self.records.get(record_id)
        if record is None:
            raise TrailerError(f"record 0x{record_id:04x} is not present in this file")
        return self._read_at(record.offset, record.length)

    def footer_mismatches(self) -> List[str]:
        """Cross-check every directory entry against the record's own footer.

        Returns a list of human-readable complaints; empty means the trailer is
        internally consistent.  Used by ``--inspect`` and as a soft warning during
        conversion, since the directory alone is enough to read the data.
        """
        problems: List[str] = []
        for record in self:
            if record.id == REC_DIRECTORY:
                continue
            end = record.offset + record.length
            if end + FOOTER_LEN > self.file_size:
                problems.append(f"0x{record.id:04x}: no room for a footer")
                continue
            footer_id, footer_len = self._unpack_footer(end)
            if footer_id != record.id or footer_len != record.length:
                problems.append(
                    f"0x{record.id:04x}: directory says (id=0x{record.id:04x}, "
                    f"len={record.length}) but footer says (id=0x{footer_id:04x}, "
                    f"len={footer_len})"
                )
        return problems
