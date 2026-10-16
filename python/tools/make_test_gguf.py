"""Generate a small synthetic GGUF (glm5_next) with known tensors for reader tests.

Writes:
  <out>/test.gguf          - the synthetic model
  <out>/test.expected.txt  - per-tensor dequantized floats (reference)

The IQ3_XXS codebook tables are parsed directly from the C++ header so the Python
reference and the C++ runtime use byte-identical tables. No third-party deps.
"""
from __future__ import annotations

import argparse
import os
import re
import struct
import random

F32, F16, Q8_0, BF16, IQ3_XXS = 0, 1, 8, 30, 18
QK_K = 256


def load_tables(header_path: str):
    text = open(header_path, "r", encoding="utf-8").read()

    def array_body(name):
        m = re.search(name + r"\s*\[[^\]]*\]\s*=\s*\{(.*?)\};", text, re.S)
        if not m:
            raise SystemExit(f"could not find array {name} in {header_path}")
        return [t.strip() for t in m.group(1).split(",") if t.strip()]

    grid = [int(t, 0) for t in array_body("kIq3xxsGrid")]
    signs = [int(t, 0) for t in array_body("kSignsIq2xs")]
    mask = [int(t, 0) for t in array_body("kMaskIq2xs")]
    assert len(grid) == 256 and len(signs) == 128 and len(mask) == 8
    return grid, signs, mask


def f32_to_half_bytes(x: float) -> bytes:
    return struct.pack("<e", x)


def f32_bytes(xs):
    return b"".join(struct.pack("<f", x) for x in xs)


def bf16_bytes(xs):
    out = bytearray()
    for x in xs:
        (bits,) = struct.unpack("<I", struct.pack("<f", x))
        out += struct.pack("<H", (bits >> 16) & 0xFFFF)
    return bytes(out)


def deq_f32(b):
    return list(struct.unpack("<%df" % (len(b) // 4), b))


def deq_f16(b):
    return list(struct.unpack("<%de" % (len(b) // 2), b))


def deq_bf16(b):
    out = []
    for i in range(0, len(b), 2):
        (h,) = struct.unpack("<H", b[i:i + 2])
        (f,) = struct.unpack("<f", struct.pack("<I", h << 16))
        out.append(f)
    return out


def deq_q8_0(b):
    out = []
    for off in range(0, len(b), 34):
        (d,) = struct.unpack("<e", b[off:off + 2])
        for q in b[off + 2:off + 34]:
            out.append(d * (q - 256 if q > 127 else q))
    return out


def deq_iq3_xxs(b, grid, signs, mask):
    out = []
    for off in range(0, len(b), 98):
        (d,) = struct.unpack("<e", b[off:off + 2])
        qs = b[off + 2:off + 98]
        ss = qs[64:]
        for ib32 in range(8):
            aux32 = struct.unpack("<I", ss[4 * ib32:4 * ib32 + 4])[0]
            qi = qs[8 * ib32:8 * ib32 + 8]
            db = d * (0.5 + (aux32 >> 28)) * 0.5
            for l in range(4):
                s = signs[(aux32 >> (7 * l)) & 127]
                g1 = struct.pack("<I", grid[qi[2 * l + 0]])
                g2 = struct.pack("<I", grid[qi[2 * l + 1]])
                grp = [0.0] * 8
                for j in range(4):
                    grp[j + 0] = db * g1[j] * (-1.0 if (s & mask[j + 0]) else 1.0)
                    grp[j + 4] = db * g2[j] * (-1.0 if (s & mask[j + 4]) else 1.0)
                out.extend(grp)
    return out


def gguf_string(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def meta_u32(k, v):
    return gguf_string(k) + struct.pack("<I", 4) + struct.pack("<I", v)


def meta_f32(k, v):
    return gguf_string(k) + struct.pack("<I", 6) + struct.pack("<f", v)


def meta_str(k, v):
    return gguf_string(k) + struct.pack("<I", 8) + gguf_string(v)


def meta_u32_array(k, vs):
    body = gguf_string(k) + struct.pack("<I", 9) + struct.pack("<I", 4) + struct.pack("<Q", len(vs))
    for v in vs:
        body += struct.pack("<I", v)
    return body


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", ".."))
    default_hdr = os.path.join(repo, "cpp", "core", "include", "glm5", "iq3xxs_tables.h")

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(repo, "build", "test"))
    ap.add_argument("--tables", default=default_hdr)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    grid, signs, mask = load_tables(args.tables)
    rng = random.Random(args.seed)

    f32_b = f32_bytes([1.5, -2.25, 3.0, 4.5, 0.0, 7.75, -0.5, 100.0])
    h16_b = b"".join(f32_to_half_bytes(x) for x in [0.5, -1.0, 2.0, -4.0])
    bf16_b = bf16_bytes([0.5, -1.0, 2.0, -4.0])

    q8_b = f32_to_half_bytes(0.125) + bytes(rng.randint(0, 255) for _ in range(32))
    iq_b = f32_to_half_bytes(0.0625) + bytes(rng.randint(0, 255) for _ in range(96))

    tensors = [
        ("test.f32", [4, 2], F32, f32_b, deq_f32(f32_b)),
        ("test.f16", [4], F16, h16_b, deq_f16(h16_b)),
        ("test.bf16", [4], BF16, bf16_b, deq_bf16(bf16_b)),
        ("test.q8_0", [32], Q8_0, q8_b, deq_q8_0(q8_b)),
        ("test.iq3_xxs", [256], IQ3_XXS, iq_b, deq_iq3_xxs(iq_b, grid, signs, mask)),
    ]

    alignment = 32
    kvs = [
        meta_str("general.architecture", "glm5_next"),
        meta_str("general.name", "glm5-selftest"),
        meta_u32("general.alignment", alignment),
        meta_u32("glm5_next.block_count", 45),
        meta_u32("glm5_next.expert_count", 288),
        meta_f32("test.float", 1.5),
        meta_u32_array("test.array", [1, 2, 3]),
    ]

    header = bytearray(b"GGUF")
    header += struct.pack("<I", 3)
    header += struct.pack("<Q", len(tensors))
    header += struct.pack("<Q", len(kvs))
    for kv in kvs:
        header += kv

    data = bytearray()
    infos = []
    for name, dims, dtype, payload, _ in tensors:
        while len(data) % alignment != 0:
            data += b"\x00"
        infos.append((name, dims, dtype, len(data)))
        data += payload

    for name, dims, dtype, offset in infos:
        header += gguf_string(name)
        header += struct.pack("<I", len(dims))
        for d in dims:
            header += struct.pack("<Q", d)
        header += struct.pack("<I", dtype)
        header += struct.pack("<Q", offset)

    while len(header) % alignment != 0:
        header += b"\x00"

    gguf_path = os.path.join(args.out, "test.gguf")
    with open(gguf_path, "wb") as f:
        f.write(header)
        f.write(data)

    exp_path = os.path.join(args.out, "test.expected.txt")
    with open(exp_path, "w", encoding="utf-8") as f:
        for name, dims, dtype, payload, expected in tensors:
            f.write(name + "\n")
            f.write(" ".join(repr(float(v)) for v in expected) + "\n")

    print(f"wrote {gguf_path} ({os.path.getsize(gguf_path)} bytes)")
    print(f"wrote {exp_path}")


if __name__ == "__main__":
    main()
