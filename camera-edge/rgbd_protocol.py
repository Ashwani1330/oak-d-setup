#!/usr/bin/env python3
from __future__ import annotations

import struct
import zlib


MAGIC_DPT = b"DPT0"
MAGIC_CAL = b"CAL0"
MAGIC_RGB = b"RGB0"

HDR_DPT = struct.Struct("<4sIHHQHHBH")
HDR_CAL = struct.Struct("<4sBQHH")
HDR_RGB = struct.Struct("<4sIQHH")
CAL_FLOATS = struct.Struct("<30fB")

COMP_NONE = 0
COMP_ZSTD = 1
COMP_LZ4 = 2
COMP_ZLIB = 3

UNITS_CM = 1
UNITS_M = 2


def compression_name(comp_id: int) -> str:
    mapping = {
        COMP_NONE: "none",
        COMP_ZSTD: "zstd",
        COMP_LZ4: "lz4",
        COMP_ZLIB: "zlib",
    }
    return mapping.get(int(comp_id), f"unknown:{comp_id}")


class DepthCompressor:
    def __init__(self, mode: str):
        self.mode = str(mode)
        self._zstd_c = None
        self._lz4 = None

        if self.mode == "zstd":
            try:
                import zstandard as zstd
                self._zstd_c = zstd.ZstdCompressor(level=1)
            except ImportError:
                raise RuntimeError("zstandard (zstd) is not installed but was requested by the profile.")

        elif self.mode == "lz4":
            try:
                import lz4.frame
                self._lz4 = lz4.frame
            except ImportError:
                raise RuntimeError("lz4 is not installed but was requested by the profile.")

        elif self.mode != "zlib":
            self.mode = "none"

    def compress(self, raw: bytes) -> tuple[bytes, int]:
        if self.mode == "none":
            return raw, COMP_NONE
        if self.mode == "zstd" and self._zstd_c is not None:
            return self._zstd_c.compress(raw), COMP_ZSTD
        if self.mode == "lz4" and self._lz4 is not None:
            return self._lz4.compress(raw, compression_level=0), COMP_LZ4
        if self.mode == "zlib":
            return zlib.compress(raw, level=1), COMP_ZLIB
        return raw, COMP_NONE


def make_decompressor():
    zstd_d = None
    lz4mod = None
    try:
        import zstandard as zstd

        zstd_d = zstd.ZstdDecompressor()
    except Exception:
        pass
    try:
        import lz4.frame

        lz4mod = lz4.frame
    except Exception:
        pass
    return zstd_d, lz4mod


def decompress_blob(blob: bytes, comp: int, *, expected: int, zstd_d, lz4mod) -> bytes:
    if comp == COMP_NONE:
        return blob
    if comp == COMP_ZSTD and zstd_d is not None:
        return zstd_d.decompress(blob, max_output_size=expected)
    if comp == COMP_LZ4 and lz4mod is not None:
        return lz4mod.decompress(blob)
    if comp == COMP_ZLIB:
        return zlib.decompress(blob)
    raise RuntimeError(f"Unsupported compression {comp} (missing lib?)")
