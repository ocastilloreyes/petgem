"""Writes reference/empymod_golden.json for the fm.tdcsem half-space test.

empymod step-on response (signal=1 for E, signal=0 times mu0 for dB/dt) of the
model in mesh.geo, sigmas.txt, sources.txt and receivers.txt. Ex keeps the
empymod sign; dBy/dt is negated.

Usage (needs empymod):
    python3 tests/tdcsem/make_reference.py
"""
import json
from pathlib import Path

import empymod
import numpy as np

MU0 = 4.0e-7 * np.pi
TIMES = np.logspace(-5, -2, 31)
OFFSETS = [200.0, 300.0, 400.0]
DEPTH = 1.0


def response(x, ab, signal):
    return empymod.dipole(src=[0.0, 0.0, DEPTH], rec=[x, 0.0, DEPTH], depth=[0.0], res=[1.0e8, 10.0],
                          freqtime=TIMES, signal=signal, ab=ab, verb=0)


golden = {
    "empymod_version": empymod.__version__,
    "times": TIMES.tolist(),
    "offsets": OFFSETS,
    "Ex": [response(x, 11, 1).tolist() for x in OFFSETS],
    "dBydt": [(-MU0 * response(x, 51, 0)).tolist() for x in OFFSETS],
}
out = Path(__file__).resolve().parent / "reference" / "empymod_golden.json"
out.write_text(json.dumps(golden, indent=1) + "\n")
print(f"wrote {out}")
