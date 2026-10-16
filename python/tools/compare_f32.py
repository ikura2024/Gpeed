"""Compare two raw float32 dumps (C++ vs Python) and report the max diff."""

import sys

import numpy as np


def main() -> int:
    a = np.fromfile(sys.argv[1], dtype=np.float32)
    b = np.fromfile(sys.argv[2], dtype=np.float32)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if n == 0:
        print("empty")
        return 1
    diff = np.abs(a - b)
    denom = np.maximum(np.abs(a), 1e-12)
    rel = diff / denom
    print(f"n={n} max_abs={diff.max():.3e} max_rel={rel.max():.3e} "
          f"mismatch(>1e-4)={(diff > 1e-4).sum()}")
    if diff.max() > 1e-4:
        i = int(diff.argmax())
        print(f"  worst at {i}: cpp={a[i]:.7g} py={b[i]:.7g}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
