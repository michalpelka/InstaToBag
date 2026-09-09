"""Minimal protobuf wire-format reader.

Insta360 stores its per-file metadata record as a serialised protobuf message but
publishes no ``.proto`` schema, so field *names* are unknowable and fields have to be
addressed by number.  Rather than depend on ``protobuf`` to parse an unknown message,
we walk the wire format directly -- it is a few dozen lines and keeps the tool
dependency-free.

Wire format reference: https://protobuf.dev/programming-guides/encoding/
"""

from __future__ import annotations

import struct
from typing import Dict, Iterator, List, Optional, Tuple

WIRE_VARINT = 0
WIRE_FIXED64 = 1
WIRE_BYTES = 2
WIRE_FIXED32 = 5

_MAX_VARINT_SHIFT = 63


class ProtobufError(ValueError):
    """Raised when a buffer is not decodable as protobuf wire format."""


def _read_varint(buf: bytes, pos: int) -> Tuple[int, int]:
    value = 0
    shift = 0
    while True:
        if pos >= len(buf):
            raise ProtobufError("truncated varint")
        byte = buf[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, pos
        shift += 7
        if shift > _MAX_VARINT_SHIFT:
            raise ProtobufError("varint longer than 64 bits")


class Message:
    """A decoded protobuf message, addressable by field number.

    Repeated fields keep every occurrence; the ``*_at`` accessors below return the
    first one, which is what every field we care about in Insta360 metadata is.
    """

    __slots__ = ("fields",)

    def __init__(self, fields: Dict[int, List[Tuple[int, object]]]) -> None:
        self.fields = fields

    @classmethod
    def parse(cls, buf: bytes) -> "Message":
        fields: Dict[int, List[Tuple[int, object]]] = {}
        pos = 0
        end = len(buf)
        while pos < end:
            key, pos = _read_varint(buf, pos)
            field_number, wire_type = key >> 3, key & 0x07
            if field_number == 0:
                raise ProtobufError("field number 0 is not valid")
            if wire_type == WIRE_VARINT:
                value, pos = _read_varint(buf, pos)
            elif wire_type == WIRE_FIXED64:
                if pos + 8 > end:
                    raise ProtobufError("truncated 64-bit field")
                value = buf[pos : pos + 8]
                pos += 8
            elif wire_type == WIRE_FIXED32:
                if pos + 4 > end:
                    raise ProtobufError("truncated 32-bit field")
                value = buf[pos : pos + 4]
                pos += 4
            elif wire_type == WIRE_BYTES:
                length, pos = _read_varint(buf, pos)
                if pos + length > end:
                    raise ProtobufError("truncated length-delimited field")
                value = buf[pos : pos + length]
                pos += length
            else:
                # Groups (3, 4) were removed from proto3 and 6/7 are unassigned.
                raise ProtobufError(f"unsupported wire type {wire_type}")
            fields.setdefault(field_number, []).append((wire_type, value))
        return cls(fields)

    def __contains__(self, field_number: int) -> bool:
        return field_number in self.fields

    def _first(self, field_number: int, wire_type: int) -> Optional[object]:
        for wt, value in self.fields.get(field_number, ()):
            if wt == wire_type:
                return value
        return None

    def uint(self, field_number: int, default: Optional[int] = None) -> Optional[int]:
        value = self._first(field_number, WIRE_VARINT)
        return default if value is None else int(value)  # type: ignore[arg-type]

    def double(self, field_number: int, default: Optional[float] = None) -> Optional[float]:
        raw = self._first(field_number, WIRE_FIXED64)
        if raw is None:
            return default
        return struct.unpack("<d", raw)[0]  # type: ignore[arg-type]

    def float(self, field_number: int, default: Optional[float] = None) -> Optional[float]:
        raw = self._first(field_number, WIRE_FIXED32)
        if raw is None:
            return default
        return struct.unpack("<f", raw)[0]  # type: ignore[arg-type]

    def raw(self, field_number: int) -> Optional[bytes]:
        value = self._first(field_number, WIRE_BYTES)
        return None if value is None else bytes(value)  # type: ignore[arg-type]

    def text(self, field_number: int, default: Optional[str] = None) -> Optional[str]:
        raw = self.raw(field_number)
        if raw is None:
            return default
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return default

    def message(self, field_number: int) -> Optional["Message"]:
        raw = self.raw(field_number)
        if raw is None:
            return None
        try:
            return Message.parse(raw)
        except ProtobufError:
            return None

    def doubles(self, field_number: int) -> List[float]:
        """Decode a length-delimited field as a packed array of float64."""
        raw = self.raw(field_number)
        if raw is None or len(raw) % 8:
            return []
        return list(struct.unpack("<%dd" % (len(raw) // 8), raw))

    def field_numbers(self) -> Iterator[int]:
        return iter(sorted(self.fields))
