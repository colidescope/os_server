from __future__ import annotations
from typing import Callable, List

TOL = 0.01

def require(cond: bool, msg: str):
    if not cond:
        raise ValueError(msg)

def unique_floats(values: List[float], tol: float) -> List[float]:
    out: List[float] = []
    for v in values:
        if not any(abs(v - u) < tol for u in out):
            out.append(v)
    out.sort()
    return out

def canonicalize_dir_world(dx: float, dy: float, dz: float, eps: float = 1e-12, shift: bool = False):
    # make a consistent sign choice

    if shift:
        if abs(dz) > eps:
            s = 1.0 if dz > 0 else -1.0
        elif abs(dx) > eps:
            s = 1.0 if dx > 0 else -1.0
        elif abs(dy) > eps:
            s = 1.0 if dy > 0 else -1.0
        else:
            s = 1.0
        return dx * s, dy * s, dz * s
    else:
        if abs(dx) > eps:
            s = 1.0 if dx > 0 else -1.0
        elif abs(dy) > eps:
            s = 1.0 if dy > 0 else -1.0
        elif abs(dz) > eps:
            s = 1.0 if dz > 0 else -1.0
        else:
            s = 1.0
        return dx * s, dy * s, dz * s