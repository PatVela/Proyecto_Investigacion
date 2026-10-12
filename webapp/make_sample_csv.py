"""Generate an example single-lead ECG CSV for the web app.

Writes the reference CSV layout: the FIRST cell of the FIRST numeric row is the
sampling rate (Hz), the rest are signal samples. Also writes a column-format
variant. These are for testing the UI; a real recording can be exported from
CinC2017 via a .mat file or a 1xN vector.

    python webapp/make_sample_csv.py --out webapp/static/example.csv
"""

from __future__ import absolute_import

import argparse
import numpy as np

FS = 300


def synth_ecg(n, kind='N'):
    t = np.arange(n) / FS
    base = 0.05 * np.sin(2 * np.pi * 1.2 * t)
    noise = np.random.randn(n) * 0.02
    if kind == 'A':
        base += 0.15 * np.sin(2 * np.pi * 4.5 * t).clip(min=0)
    elif kind == 'O':
        base += 0.15 * np.sin(2 * np.pi * 2.0 * t).clip(min=0)
    else:
        base += 0.15 * np.sin(2 * np.pi * 1.3 * t).clip(min=0)
    return (base + noise).astype(np.float32)


def main(out, n, kind):
    sig = synth_ecg(n, kind)
    # row layout: fs, s0, s1, ...
    with open(out, 'w') as f:
        f.write("Lead,II\n")
        f.write(",".join([str(FS)] + ["{:.6f}".format(v) for v in sig]) + "\n")
    print("Escribí:", out, "| n =", n, "| fs =", FS, "| clase base =", kind)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="webapp/static/example.csv")
    p.add_argument("--n", type=int, default=256 * 30, help="muestras (múltiplo de 256)")
    p.add_argument("--kind", default="N", choices=['N', 'A', 'O', '~', '|'])
    args = p.parse_args()
    main(args.out, args.n, args.kind)
