"""Shared helpers: synthetic protobuf messages and synthetic Insta360 trailers.

The tests build real byte layouts rather than mocking the parsers, so a change to the
on-disk format understanding shows up as a test failure.
"""

from __future__ import annotations

import os
import struct
from typing import Iterable, List, Tuple

import pytest

from insta360_to_bag import trailer as trailer_mod

REAL_CAPTURE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "VID_20260909_132827_00_005.insv",
)

requires_real_capture = pytest.mark.skipif(
    not os.path.exists(REAL_CAPTURE),
    reason="sample .insv capture not present in the repository root",
)


#: Calibration string from an Insta360 X5 capture (metadata field 54), verbatim.
X5_OFFSET_V2 = (
    "2_2.000000_4299.640_4299.070_2697.020_2681.800_0.196_-0.027_89.717_"
    "0.000000_0.000000_0.000000_0.18967018_2.06612277_-3.31555128_0.00052395_"
    "0.00003870_10752_5376_113_2.000000_4273.430_4273.780_8083.250_2666.280_"
    "-0.136_-0.096_90.887_0.002375_-0.000169_-0.031969_0.19482063_1.98416793_"
    "-3.08075333_0.00059304_0.00104311_10752_5376_113_197632"
)


# -- protobuf encoding ------------------------------------------------------


def varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def pb_varint(field: int, value: int) -> bytes:
    return varint(field << 3 | 0) + varint(value)


def pb_double(field: int, value: float) -> bytes:
    return varint(field << 3 | 1) + struct.pack("<d", value)


def pb_float(field: int, value: float) -> bytes:
    return varint(field << 3 | 5) + struct.pack("<f", value)


def pb_bytes(field: int, payload: bytes) -> bytes:
    return varint(field << 3 | 2) + varint(len(payload)) + payload


def pb_string(field: int, text: str) -> bytes:
    return pb_bytes(field, text.encode("utf-8"))


# -- trailer construction ---------------------------------------------------


def build_trailer(records: Iterable[Tuple[int, bytes]], prefix: bytes = b"") -> bytes:
    """Assemble a byte-exact Insta360 trailer holding ``records``.

    Mirrors the real layout: payloads back to back, each followed by a 6-byte footer
    with a big-endian id and little-endian length, then a directory record whose
    entries are little-endian and whose offsets are relative to the trailer start,
    then the 72-byte tail.
    """
    records = list(records)
    body = bytearray()
    entries: List[Tuple[int, int, int]] = []
    for record_id, payload in records:
        entries.append((record_id, len(payload), len(body)))
        body += payload
        body += struct.pack(">H", record_id) + struct.pack("<I", len(payload))

    directory = bytearray(b"\x00" * trailer_mod.DIRECTORY_HEADER_LEN)
    for record_id, length, offset in entries:
        directory += struct.pack("<HII", record_id, length, offset)
    # Real files leave unused, zero-filled slots in the table.
    directory += b"\x00" * trailer_mod.DIRECTORY_ENTRY_LEN

    body += directory
    body += struct.pack(">H", trailer_mod.REC_DIRECTORY) + struct.pack("<I", len(directory))

    trailer_size = len(body) + trailer_mod.TAIL_LEN
    tail = b"\x00" * 32 + struct.pack("<II", trailer_size, 3) + trailer_mod.MAGIC
    return prefix + bytes(body) + tail


@pytest.fixture
def write_insv(tmp_path):
    def _write(records: Iterable[Tuple[int, bytes]], prefix: bytes = b"fake mp4 boxes") -> str:
        path = tmp_path / "sample.insv"
        path.write_bytes(build_trailer(records, prefix=prefix))
        return str(path)

    return _write
