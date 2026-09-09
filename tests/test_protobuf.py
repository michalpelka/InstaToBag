import struct

import pytest

from insta360_to_bag.protobuf import Message, ProtobufError

from conftest import pb_bytes, pb_double, pb_float, pb_string, pb_varint


def test_reads_each_wire_type():
    raw = (
        pb_varint(1, 300)
        + pb_string(2, "Insta360 X5")
        + pb_double(3, 1.6)
        + pb_float(4, 0.5)
        + pb_bytes(5, b"\x00\x01\xff")
    )
    message = Message.parse(raw)
    assert message.uint(1) == 300
    assert message.text(2) == "Insta360 X5"
    assert message.double(3) == pytest.approx(1.6)
    assert message.float(4) == pytest.approx(0.5)
    assert message.raw(5) == b"\x00\x01\xff"


def test_absent_fields_return_the_default():
    message = Message.parse(pb_varint(1, 7))
    assert message.uint(99) is None
    assert message.uint(99, 5) == 5
    assert message.text(99, "fallback") == "fallback"
    assert message.message(99) is None
    assert 1 in message and 99 not in message


def test_nested_messages_decode():
    inner = pb_varint(1, 2880) + pb_varint(2, 2880)
    message = Message.parse(pb_bytes(19, inner))
    nested = message.message(19)
    assert nested is not None
    assert (nested.uint(1), nested.uint(2)) == (2880, 2880)


def test_repeated_fields_keep_the_first():
    message = Message.parse(pb_varint(1, 10) + pb_varint(1, 20))
    assert message.uint(1) == 10
    assert len(message.fields[1]) == 2


def test_wrong_wire_type_does_not_leak_across_accessors():
    message = Message.parse(pb_string(1, "text"))
    assert message.uint(1) is None
    assert message.text(1) == "text"


def test_packed_doubles():
    payload = struct.pack("<3d", 1.0, 2.0, 3.0)
    message = Message.parse(pb_bytes(31, payload))
    assert message.doubles(31) == [1.0, 2.0, 3.0]
    # A length that is not a multiple of 8 is not a double array.
    assert Message.parse(pb_bytes(31, b"\x00" * 7)).doubles(31) == []


def test_non_utf8_text_falls_back_instead_of_raising():
    message = Message.parse(pb_bytes(1, b"\xff\xfe"))
    assert message.text(1) is None
    assert message.text(1, "?") == "?"


@pytest.mark.parametrize(
    "raw",
    [
        b"\x08",  # truncated varint
        b"\x09\x00",  # truncated fixed64
        b"\x0d\x00",  # truncated fixed32
        b"\x0a\x05ab",  # length-delimited field runs past the end
        b"\x00\x01",  # field number zero
        b"\x1c",  # wire type 4 (removed group end)
    ],
)
def test_malformed_input_raises(raw):
    with pytest.raises(ProtobufError):
        Message.parse(raw)


def test_overlong_varint_is_rejected():
    with pytest.raises(ProtobufError):
        Message.parse(b"\x08" + b"\xff" * 12)
