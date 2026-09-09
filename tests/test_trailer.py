import struct

import pytest

from insta360_to_bag import trailer as trailer_mod

from conftest import REAL_CAPTURE, build_trailer, requires_real_capture


def test_reads_records_from_a_synthetic_trailer(write_insv):
    path = write_insv([
        (trailer_mod.REC_METADATA, b"metadata payload"),
        (trailer_mod.REC_IMU, b"\x01" * 40),
        (trailer_mod.REC_EXPOSURE, b"\x02" * 32),
    ])
    with trailer_mod.Trailer(path) as trailer:
        assert set(trailer.records) == {
            trailer_mod.REC_METADATA,
            trailer_mod.REC_IMU,
            trailer_mod.REC_EXPOSURE,
        }
        assert trailer.read(trailer_mod.REC_METADATA) == b"metadata payload"
        assert trailer.read(trailer_mod.REC_IMU) == b"\x01" * 40
        assert trailer.version == 3
        assert trailer.footer_mismatches() == []


def test_records_iterate_in_file_order(write_insv):
    path = write_insv([(0x0003, b"a" * 8), (0x0004, b"b" * 8), (0x0101, b"c" * 8)])
    with trailer_mod.Trailer(path) as trailer:
        offsets = [record.offset for record in trailer]
        assert offsets == sorted(offsets)
        assert [record.id for record in trailer] == [0x0003, 0x0004, 0x0101]


def test_zero_padded_directory_slots_are_ignored(write_insv):
    # build_trailer always appends one all-zero slot, mimicking real firmware output.
    path = write_insv([(trailer_mod.REC_IMU, b"x" * 20)])
    with trailer_mod.Trailer(path) as trailer:
        assert list(trailer.records) == [trailer_mod.REC_IMU]


def test_unknown_record_ids_are_still_readable(write_insv):
    path = write_insv([(0x4321, b"mystery")])
    with trailer_mod.Trailer(path) as trailer:
        assert trailer.read(0x4321) == b"mystery"
        assert "unknown 0x4321" in trailer.records[0x4321].name


def test_missing_magic_is_rejected(tmp_path):
    path = tmp_path / "plain.mp4"
    path.write_bytes(b"\x00" * 4096)
    with pytest.raises(trailer_mod.TrailerError, match="magic not found"):
        trailer_mod.Trailer(str(path))


def test_tiny_file_is_rejected(tmp_path):
    path = tmp_path / "tiny.insv"
    path.write_bytes(b"short")
    with pytest.raises(trailer_mod.TrailerError, match="too small"):
        trailer_mod.Trailer(str(path))


def test_implausible_trailer_size_is_rejected(tmp_path):
    tail = b"\x00" * 32 + struct.pack("<II", 1 << 30, 3) + trailer_mod.MAGIC
    path = tmp_path / "bad.insv"
    path.write_bytes(b"\x00" * 1024 + tail)
    with pytest.raises(trailer_mod.TrailerError, match="implausible trailer size"):
        trailer_mod.Trailer(str(path))


def test_directory_entry_pointing_outside_the_file_is_rejected(tmp_path):
    raw = bytearray(build_trailer([(trailer_mod.REC_IMU, b"y" * 16)]))
    # The directory's single entry ends with a 4-byte offset; push it past EOF.
    entry_offset_at = raw.rindex(struct.pack("<HII", trailer_mod.REC_IMU, 16, 0)) + 6
    raw[entry_offset_at : entry_offset_at + 4] = struct.pack("<I", 1 << 28)
    path = tmp_path / "corrupt.insv"
    path.write_bytes(bytes(raw))
    with pytest.raises(trailer_mod.TrailerError, match="outside the file"):
        trailer_mod.Trailer(str(path))


def test_reading_an_absent_record_raises(write_insv):
    path = write_insv([(trailer_mod.REC_IMU, b"z" * 20)])
    with trailer_mod.Trailer(path) as trailer:
        with pytest.raises(trailer_mod.TrailerError, match="not present"):
            trailer.read(trailer_mod.REC_METADATA)


def test_footer_mismatch_is_reported_not_raised(tmp_path):
    raw = bytearray(build_trailer([(trailer_mod.REC_IMU, b"w" * 16)]))
    # Corrupt the record's own footer length; the directory still reads fine.
    footer_at = raw.index(b"w" * 16) + 16
    raw[footer_at + 2 : footer_at + 6] = struct.pack("<I", 999)
    path = tmp_path / "mismatch.insv"
    path.write_bytes(bytes(raw))
    with trailer_mod.Trailer(str(path)) as trailer:
        assert trailer.read(trailer_mod.REC_IMU) == b"w" * 16
        problems = trailer.footer_mismatches()
        assert len(problems) == 1 and "len=999" in problems[0]


@requires_real_capture
def test_real_capture_layout():
    with trailer_mod.Trailer(REAL_CAPTURE) as trailer:
        assert trailer.version == 3
        assert trailer.footer_mismatches() == []
        for record_id in (
            trailer_mod.REC_METADATA,
            trailer_mod.REC_IMU,
            trailer_mod.REC_EXPOSURE,
            trailer_mod.REC_PREVIEW,
        ):
            assert record_id in trailer
        # Metadata field 9 restates where the trailer begins; a good cross-check that
        # the tail's size field was read correctly.
        from insta360_to_bag.protobuf import Message
        message = Message.parse(trailer.read(trailer_mod.REC_METADATA))
        assert message.uint(9) == trailer.base_offset
