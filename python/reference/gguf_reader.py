"""Minimal, dependency-light GGUF reader (reference side).

Independent from the C++ implementation on purpose: it is used to cross-check
the C++ parser and the dequantization codecs on real model shards.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Any


# GGUF metadata value types.
UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL, STRING, ARRAY, \
    UINT64, INT64, FLOAT64 = range(13)

_SCALAR_FMT = {
    UINT8: "<B", INT8: "<b", UINT16: "<H", INT16: "<h", UINT32: "<I",
    INT32: "<i", FLOAT32: "<f", BOOL: "<?", UINT64: "<Q", INT64: "<q",
    FLOAT64: "<d",
}

# ggml_type enum values (subset we care about).
GGML_TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
    19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
    24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
    30: "BF16", 34: "TQ1_0", 35: "TQ2_0",
}


@dataclass
class TensorInfo:
    name: str
    dims: tuple[int, ...]
    ggml_type: int
    offset: int

    @property
    def type_name(self) -> str:
        return GGML_TYPE_NAMES.get(self.ggml_type, f"?{self.ggml_type}")

    @property
    def n_elements(self) -> int:
        n = 1
        for d in self.dims:
            n *= d
        return n


@dataclass
class GgufReader:
    path: str
    version: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    tensors: dict[str, TensorInfo] = field(default_factory=dict)
    tensor_order: list[str] = field(default_factory=list)
    data_offset: int = 0
    _fh: Any = None

    @classmethod
    def open(cls, path: str) -> "GgufReader":
        self = cls(path=path)
        self._fh = open(path, "rb")
        self._parse()
        return self

    def _parse(self) -> None:
        f = self._fh
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"not a GGUF file: {magic!r}")
        (self.version,) = struct.unpack("<I", f.read(4))
        if self.version not in (2, 3):
            raise ValueError(f"unsupported GGUF version {self.version}")
        (n_tensors,) = struct.unpack("<Q", f.read(8))
        (n_kv,) = struct.unpack("<Q", f.read(8))

        for _ in range(n_kv):
            key = self._read_string()
            self.metadata[key] = self._read_value()

        for _ in range(n_tensors):
            name = self._read_string()
            (nd,) = struct.unpack("<I", f.read(4))
            dims = struct.unpack(f"<{nd}Q", f.read(8 * nd)) if nd else ()
            (ttype,) = struct.unpack("<I", f.read(4))
            (off,) = struct.unpack("<Q", f.read(8))
            info = TensorInfo(name, tuple(dims), ttype, off)
            self.tensors[name] = info
            self.tensor_order.append(name)

        alignment = self.metadata.get("general.alignment", 32)
        pos = f.tell()
        self.data_offset = (pos + alignment - 1) // alignment * alignment

    def _read_string(self) -> str:
        (n,) = struct.unpack("<Q", self._fh.read(8))
        return self._fh.read(n).decode("utf-8")

    def _read_value(self, vt: int | None = None) -> Any:
        if vt is None:
            (vt,) = struct.unpack("<I", self._fh.read(4))
        if vt == STRING:
            return self._read_string()
        if vt == ARRAY:
            (etype,) = struct.unpack("<I", self._fh.read(4))
            (n,) = struct.unpack("<Q", self._fh.read(8))
            return [self._read_value(etype) for _ in range(n)]
        fmt = _SCALAR_FMT[vt]
        return struct.unpack(fmt, self._fh.read(struct.calcsize(fmt)))[0]

    def read_tensor(self, name: str, nbytes: int) -> bytes:
        info = self.tensors[name]
        self._fh.seek(self.data_offset + info.offset)
        data = self._fh.read(nbytes)
        if len(data) != nbytes:
            raise IOError(f"short read for {name}: {len(data)} != {nbytes}")
        return data


def discover_shards(any_shard: str) -> list[str]:
    """Glob '*-NNNNN-of-MMMMM.gguf' siblings given any one shard path."""
    import glob
    import re

    m = re.search(r"-\d{5}-of-\d{5}\.gguf$", any_shard)
    if not m:
        return [any_shard]
    base = any_shard[: m.start()]
    files = sorted(glob.glob(base + "-*-of-*.gguf"))
    return files or [any_shard]


def open_model(any_shard: str) -> tuple[list[GgufReader], dict[str, tuple[GgufReader, TensorInfo]]]:
    """Open every shard and return (readers, {tensor_name: (shard_reader, info)})."""
    readers = [GgufReader.open(p) for p in discover_shards(any_shard)]
    tensors: dict[str, tuple[GgufReader, TensorInfo]] = {}
    for r in readers:
        for name, info in r.tensors.items():
            tensors[name] = (r, info)
    return readers, tensors
