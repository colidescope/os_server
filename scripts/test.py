# bracket/bracket_script.py
from __future__ import annotations
import math

from typing import Any, Callable, Dict, List, Optional, Tuple

from helpers import log, require
from geom.common import TOL
from geom.protocol import GeomBackend, GCircle, GCurve, GLine, GPlane, GPoint, GVector, GSolid, Interval

DEFAULT_TOL = TOL

def main(
    geom: GeomBackend,
) -> Dict[str, Any]:
    """
    Test script.
    """

    pt = geom.point(1, 1, 5)
    log(type(pt))
            
    return {
        "pt": pt.translated(geom.vector(1,0,0))._serialize(),
    }