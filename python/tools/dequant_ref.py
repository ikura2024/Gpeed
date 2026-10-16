"""Independent reference dequantizer, cross-checked against the C++ codecs.

The lookup tables are parsed straight out of the C++ headers so that this file
only validates the *logic* (bit unpacking / pointer walking), not the data.
Usage:
    python dequant_ref.py <any-shard.gguf> <tensor-name> <max-elements> [out.f32]
"""

from __future__ import annotations

import os
import re
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "reference"))
from gguf_reader import GgufReader, open_model  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HDR_TABLES = os.path.join(ROOT, "cpp", "core", "include", "glm5", "quant_tables.h")
HDR_IQ3XXS = os.path.join(ROOT, "cpp", "core", "include", "glm5", "iq3xxs_tables.h")


def _parse_array(text: str, name: str, dtype) -> np.ndarray:
    m = re.search(name + r"\s*\[[^\]]*\]\s*=\s*\{(.*?)\};", text, re.S)
    if not m:
        raise KeyError(name)
    body = re.sub(r"/\*.*?\*/", "", m.group(1), flags=re.S)
    body = re.sub(r"//[^\n]*", "", body)
    toks = [t for t in re.split(r"[,\s]+", body) if t]
    return np.array([int(t, 0) for t in toks], dtype=dtype)


_tab = open(HDR_TABLES, "r", encoding="utf-8").read()
_xxs = open(HDR_IQ3XXS, "r", encoding="utf-8").read()

MASK = _parse_array(_xxs, "kMaskIq2xs", np.uint8)
IQ2S_GRID = _parse_array(_tab, "kIq2sGrid", np.uint64)
IQ3S_GRID = _parse_array(_tab, "kIq3sGrid", np.uint32)
IQ4NL = _parse_array(_tab, "kValuesIq4nl", np.int8).astype(np.float32)
IQ3XXS_GRID = _parse_array(_xxs, "kIq3xxsGrid", np.uint32)
SIGNS_IQ2XS = _parse_array(_xxs, "kSignsIq2xs", np.uint8)

IQ2S_U8 = IQ2S_GRID.view(np.uint8).reshape(-1, 8)
IQ3S_U8 = IQ3S_GRID.view(np.uint8).reshape(-1, 4)
IQ3XXS_U8 = IQ3XXS_GRID.view(np.uint8).reshape(-1, 4)


def _halfs(raw: bytes, stride: int, off: int, nb: int) -> np.ndarray:
    a = np.frombuffer(raw, np.uint8).reshape(nb, stride)
    return a[:, off:off + 2].copy().reshape(nb, 2).view("<f2")[:, 0].astype(np.float32)


def deq_f32(raw, nb, n):
    return np.frombuffer(raw, np.float32, count=n).copy()


def deq_f16(raw, nb, n):
    return np.frombuffer(raw, "<f2", count=n).astype(np.float32)


def deq_bf16(raw, nb, n):
    u = np.frombuffer(raw, np.uint16, count=n).astype(np.uint32) << 16
    return u.view(np.float32).copy()


def deq_q8_0(raw, nb, n):
    a = np.frombuffer(raw, np.uint8).reshape(nb, 34)
    d = a[:, 0:2].copy().reshape(nb, 2).view("<f2")[:, 0].astype(np.float32)
    q = a[:, 2:34].view(np.int8).astype(np.float32)
    return (d[:, None] * q).reshape(-1)


def deq_q6_K(raw, nb, n):
    a = np.frombuffer(raw, np.uint8).reshape(nb, 210)
    out = np.empty((nb, 256), np.float32)
    for b in range(nb):
        ql = a[b, 0:128].astype(np.int32)
        qh = a[b, 128:192].astype(np.int32)
        sc = a[b, 192:208].view(np.int8).astype(np.int32)
        d = struct.unpack_from("<e", a[b].tobytes(), 208)[0]
        y = out[b]
        for blk in range(2):
            o = blk * 128
            qlb = ql[blk * 64: blk * 64 + 64]
            qhb = qh[blk * 32: blk * 32 + 32]
            scb = sc[blk * 8: blk * 8 + 8]
            for l in range(32):
                is_ = l // 16
                q1 = ((qlb[l] & 0xF) | (((qhb[l] >> 0) & 3) << 4)) - 32
                q2 = ((qlb[l + 32] & 0xF) | (((qhb[l] >> 2) & 3) << 4)) - 32
                q3 = ((qlb[l] >> 4) | (((qhb[l] >> 4) & 3) << 4)) - 32
                q4 = ((qlb[l + 32] >> 4) | (((qhb[l] >> 6) & 3) << 4)) - 32
                y[o + l + 0] = d * scb[is_ + 0] * q1
                y[o + l + 32] = d * scb[is_ + 2] * q2
                y[o + l + 64] = d * scb[is_ + 4] * q3
                y[o + l + 96] = d * scb[is_ + 6] * q4
    return out.reshape(-1)


def deq_q2_K(raw, nb, n):
    a = np.frombuffer(raw, np.uint8).reshape(nb, 84)
    out = np.empty((nb, 256), np.float32)
    for b in range(nb):
        sc = a[b, 0:16]
        q = a[b, 16:80]
        d = struct.unpack_from("<e", a[b].tobytes(), 80)[0]
        dm = struct.unpack_from("<e", a[b].tobytes(), 82)[0]
        y = out[b]
        is_ = 0
        for base in (0, 128):
            shift = 0
            for _ in range(4):
                s0 = sc[is_]; is_ += 1
                dl = d * (s0 & 0xF); ml = dm * (s0 >> 4)
                for l in range(16):
                    y[base + shift // 2 * 32 + l] = dl * ((q[l] >> shift) & 3) - ml
                s1 = sc[is_]; is_ += 1
                dl = d * (s1 & 0xF); ml = dm * (s1 >> 4)
                for l in range(16):
                    y[base + shift // 2 * 32 + 16 + l] = dl * ((q[l + 16] >> shift) & 3) - ml
                shift += 2
            q = q[32:]
    return out.reshape(-1)


def deq_q3_K(raw, nb, n):
    a = np.frombuffer(raw, np.uint8).reshape(nb, 110)
    out = np.empty((nb, 256), np.float32)
    k1, k2 = 0x03030303, 0x0F0F0F0F
    for b in range(nb):
        hm = a[b, 0:32].astype(np.int32)
        q = a[b, 32:96]
        aux = np.zeros(4, np.uint32)
        aux[:3] = np.frombuffer(a[b, 96:108].tobytes(), np.uint32)
        tmp = aux[2]
        aux[2] = ((aux[0] >> 4) & k2) | (((tmp >> 4) & k1) << 4)
        aux[3] = ((aux[1] >> 4) & k2) | (((tmp >> 6) & k1) << 4)
        aux[0] = (aux[0] & k2) | (((tmp >> 0) & k1) << 4)
        aux[1] = (aux[1] & k2) | (((tmp >> 2) & k1) << 4)
        scales = aux.view(np.int8)
        d_all = struct.unpack_from("<e", a[b].tobytes(), 108)[0]
        y = out[b]
        is_ = 0
        m = 1
        for base in (0, 128):
            shift = 0
            qb = q[(base // 128) * 32:(base // 128) * 32 + 32]
            for _ in range(4):
                dl = d_all * (int(scales[is_]) - 32); is_ += 1
                for l in range(16):
                    y[base + shift // 2 * 32 + l] = dl * (
                        int((qb[l] >> shift) & 3) - (0 if (hm[l] & m) else 4))
                dl = d_all * (int(scales[is_]) - 32); is_ += 1
                for l in range(16):
                    y[base + shift // 2 * 32 + 16 + l] = dl * (
                        int((qb[l + 16] >> shift) & 3) - (0 if (hm[l + 16] & m) else 4))
                shift += 2
                m <<= 1
    return out.reshape(-1)


def deq_iq2_s(raw, nb, n):
    a = np.frombuffer(raw, np.uint8).reshape(nb, 82)
    out = np.empty((nb, 256), np.float32)
    for b in range(nb):
        d = struct.unpack_from("<e", a[b].tobytes(), 0)[0]
        qs = a[b, 2:66]
        qh = a[b, 66:74]
        sc = a[b, 74:82]
        signs = qs[32:64]
        y = out[b]
        for ib32 in range(8):
            db0 = d * (0.5 + (sc[ib32] & 0xF)) * 0.25
            db1 = d * (0.5 + (sc[ib32] >> 4)) * 0.25
            for l in range(4):
                dl = db0 if l < 2 else db1
                idx = int(qs[ib32 * 4 + l]) | ((int(qh[ib32]) << (8 - 2 * l)) & 0x300)
                g = IQ2S_U8[idx]
                sg = int(signs[ib32 * 4 + l])
                for j in range(8):
                    y[ib32 * 32 + l * 8 + j] = dl * g[j] * (-1.0 if (sg & MASK[j]) else 1.0)
    return out.reshape(-1)


def deq_iq3_s(raw, nb, n):
    a = np.frombuffer(raw, np.uint8).reshape(nb, 110)
    out = np.empty((nb, 256), np.float32)
    for b in range(nb):
        d = struct.unpack_from("<e", a[b].tobytes(), 0)[0]
        qs = a[b, 2:66]
        qh = a[b, 66:74]
        signs = a[b, 74:106]
        sc = a[b, 106:110]
        y = out[b]
        for k in range(4):
            db1 = d * (1 + 2 * (sc[k] & 0xF))
            db2 = d * (1 + 2 * (sc[k] >> 4))
            for p in range(2):
                db = db1 if p == 0 else db2
                qbase = k * 16 + p * 8
                sbase = k * 8 + p * 4
                qhv = int(qh[k * 2 + p])
                for l in range(4):
                    i1 = int(qs[qbase + 2 * l]) | ((qhv << (8 - 2 * l)) & 256)
                    i2 = int(qs[qbase + 2 * l + 1]) | ((qhv << (7 - 2 * l)) & 256)
                    g1 = IQ3S_U8[i1]
                    g2 = IQ3S_U8[i2]
                    sg = int(signs[sbase + l])
                    o = k * 64 + p * 32 + l * 8
                    for j in range(4):
                        y[o + j + 0] = db * g1[j] * (-1.0 if (sg & MASK[j]) else 1.0)
                        y[o + j + 4] = db * g2[j] * (-1.0 if (sg & MASK[j + 4]) else 1.0)
    return out.reshape(-1)


def deq_iq4_xs(raw, nb, n):
    a = np.frombuffer(raw, np.uint8).reshape(nb, 136)
    out = np.empty((nb, 256), np.float32)
    for b in range(nb):
        d = struct.unpack_from("<e", a[b].tobytes(), 0)[0]
        sh = struct.unpack_from("<H", a[b].tobytes(), 2)[0]
        sl = a[b, 4:8]
        qs = a[b, 8:136]
        y = out[b]
        for ib in range(8):
            ls = (int(sl[ib // 2] >> (4 * (ib % 2))) & 0xF) | ((int(sh >> (2 * ib)) & 3) << 4)
            dl = d * (ls - 32)
            seg = qs[ib * 16: ib * 16 + 16]
            for j in range(16):
                y[ib * 32 + j] = dl * IQ4NL[seg[j] & 0xF]
                y[ib * 32 + 16 + j] = dl * IQ4NL[seg[j] >> 4]
    return out.reshape(-1)


def deq_iq3_xxs(raw, nb, n):
    a = np.frombuffer(raw, np.uint8).reshape(nb, 98)
    out = np.empty((nb, 256), np.float32)
    for b in range(nb):
        d = struct.unpack_from("<e", a[b].tobytes(), 0)[0]
        qs = a[b, 2:98]
        sas = qs[64:96]
        y = out[b]
        for ib32 in range(8):
            aux = struct.unpack_from("<I", sas.tobytes(), 4 * ib32)[0]
            db = d * (0.5 + (aux >> 28)) * 0.5
            for l in range(4):
                sg = int(SIGNS_IQ2XS[(aux >> (7 * l)) & 127])
                g1 = IQ3XXS_U8[qs[8 * ib32 + 2 * l + 0]]
                g2 = IQ3XXS_U8[qs[8 * ib32 + 2 * l + 1]]
                for j in range(4):
                    y[ib32 * 32 + l * 8 + j] = db * g1[j] * (-1.0 if (sg & MASK[j]) else 1.0)
                    y[ib32 * 32 + l * 8 + j + 4] = db * g2[j] * (-1.0 if (sg & MASK[j + 4]) else 1.0)
    return out.reshape(-1)


TYPE_SIZE = {
    "F32": 4, "F16": 2, "BF16": 2, "Q8_0": 34, "Q6_K": 210, "Q2_K": 84, "Q3_K": 110,
    "IQ2_S": 82, "IQ3_S": 110, "IQ4_XS": 136, "IQ3_XXS": 98,
}
BLOCK = {k: 1 for k in TYPE_SIZE}
for k in TYPE_SIZE:
    if k not in ("F32", "F16", "BF16"):
        BLOCK[k] = 256

DEQ = {
    "F32": deq_f32, "F16": deq_f16, "BF16": deq_bf16, "Q8_0": deq_q8_0,
    "Q6_K": deq_q6_K, "Q2_K": deq_q2_K, "Q3_K": deq_q3_K, "IQ2_S": deq_iq2_s,
    "IQ3_S": deq_iq3_s, "IQ4_XS": deq_iq4_xs, "IQ3_XXS": deq_iq3_xxs,
}


def main() -> int:
    if len(sys.argv) < 4:
        print(__doc__)
        return 2
    path, name, max_el = sys.argv[1], sys.argv[2], int(sys.argv[3])
    out_path = sys.argv[4] if len(sys.argv) > 4 else None

    _readers, tensors = open_model(path)
    if name not in tensors:
        print(f"tensor not found: {name}")
        return 1
    r, info = tensors[name]
    tn = info.type_name
    if tn not in DEQ:
        print(f"unsupported type {tn}")
        return 1

    n = info.n_elements
    neff = min(max_el, n)
    bs = BLOCK[tn]
    nb_needed = (neff + bs - 1) // bs
    bytes_needed = nb_needed * TYPE_SIZE[tn]
    raw = r.read_tensor(name, bytes_needed)
    vals = DEQ[tn](raw, nb_needed, neff)[:neff].astype(np.float32)
    print(f"tensor      : {name}")
    print(f"type        : {tn}")
    print(f"elements    : {n} (checked {neff})")
    print(f"min         : {vals.min():.6g}")
    print(f"max         : {vals.max():.6g}")
    print(f"mean        : {vals.mean():.6g}")
    print(f"std         : {vals.std():.6g}")
    print(f"absmean     : {np.abs(vals).mean():.6g}")
    print("first8      :", " ".join(f"{v:.7g}" for v in vals[:8]))

    if out_path:
        with open(out_path, "wb") as fh:
            fh.write(vals.tobytes())
        print(f"wrote       : {neff} floats to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
