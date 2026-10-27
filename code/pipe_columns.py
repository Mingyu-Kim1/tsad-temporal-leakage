"""pipe_columns.py — channel names of the piping (sim-to-real) data.

The piping data are provided under a non-disclosure agreement and are not distributed,
and neither is the correspondence between simulation and real channels.
Place the names in  work/data/pipe/columns.json  (not tracked by git):

    {"sim": ["<sim channel 1>", ..., "<sim channel 5>"],
     "real": ["<real channel 1>", ..., "<real channel 5>"]}

Without the file the Sim2Real pair cannot be run; all other pairs are unaffected.
"""
import json
from pathlib import Path

import paths

_F = paths.DATA / "pipe" / "columns.json"


def _load():
    if _F.exists():
        c = json.loads(_F.read_text(encoding="utf-8"))
        return list(c["sim"]), list(c["real"])
    ph = [f"ch{i}" for i in range(1, 6)]
    return ph, ph


SIM_COLS, REAL_COLS = _load()
