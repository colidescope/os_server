# geom/occ_backend.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

from geom.protocol import GPoint, GVector, GPlane, GLine, GCurve, GCircle, GSolid
from geom.common import TOL, require, unique_floats, canonicalize_dir_world
from geom.occ_helpers import shape_to_mesh_payload

# ---------------------------
# pythonocc-core imports
# ---------------------------
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Face, TopoDS_Wire, TopoDS_Edge, TopoDS_Vertex
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_WIRE, TopAbs_EDGE, TopAbs_VERTEX, TopAbs_IN, TopAbs_OUT, TopAbs_ON
from OCC.Core.TopExp import TopExp_Explorer

from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeWire,
    BRepBuilderAPI_Transform,
    BRepBuilderAPI_MakeFace,
)

from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse

from OCC.Core.BRepGProp import brepgprop_VolumeProperties
from OCC.Core.GProp import GProp_GProps

from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib_Add

from OCC.Core.BRepAdaptor import BRepAdaptor_Surface

from OCC.Core.GeomAbs import GeomAbs_Plane
from OCC.Core.gp import (
    gp_Pnt,
    gp_Vec,
    gp_Dir,
    gp_Trsf,
    gp_Ax1,
    gp_Ax2,
    gp_Ax3,
    gp_Pln,
    gp_Lin,
    gp_Circ
)

from OCC.Core.GC import GC_MakeCircle
from OCC.Core.BRepTools import breptools_OuterWire

# ======================================================================
# Low-level helpers
# ======================================================================

def _gp_pnt_from_gpoint(p: GPoint) -> gp_Pnt:
    if isinstance(p, OCCPoint):
        return p._p
    return gp_Pnt(float(p.x), float(p.y), float(p.z))

def _gp_vec_from_gvec(v: GVector) -> gp_Vec:
    if isinstance(v, OCCVector):
        return v._v
    return gp_Vec(float(v.x), float(v.y), float(v.z))

def _gp_pln_from_gplane(pl: GPlane) -> gp_Pln:
    if isinstance(pl, OCCPlane):
        return pl._pln
    # If someone passes a raw gp_Pln, accept it
    if isinstance(pl, gp_Pln):
        return pl
    raise TypeError(f"Unsupported plane type: {type(pl)}")

def _vec_cross(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
    ax, ay, az = a
    bx, by, bz = b
    return (ay*bz - az*by, az*bx - ax*bz, ax*by - ay*bx)

def _vec_dot(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def _vec_len2(a: Tuple[float, float, float]) -> float:
    return _vec_dot(a, a)

def _plane_point_to_uv(pln: gp_Pln, p: gp_Pnt) -> Tuple[float, float]:
    """
    Convert a 3D point to plane-local (u,v) using the plane's Ax3 (origin + X/Y directions).
    """
    ax3: gp_Ax3 = pln.Position()
    o = ax3.Location()
    xdir = ax3.XDirection()
    ydir = ax3.YDirection()

    vx = p.X() - o.X()
    vy = p.Y() - o.Y()
    vz = p.Z() - o.Z()

    u = vx*xdir.X() + vy*xdir.Y() + vz*xdir.Z()
    v = vx*ydir.X() + vy*ydir.Y() + vz*ydir.Z()
    return float(u), float(v)


# ======================================================================
# OCC geometry wrappers
# ======================================================================

class OCCPoint:
    def __init__(self, p: gp_Pnt):
        self._p = p

    @property
    def x(self) -> float: return float(self._p.X())
    @property
    def y(self) -> float: return float(self._p.Y())
    @property
    def z(self) -> float: return float(self._p.Z())

    def __str__(self):
        return f"[{self._p.X()}, {self._p.Y()}, {self._p.Z()}]"

    def translated(self, v: "GVector") -> "OCCPoint":
        vv = _gp_vec_from_gvec(v)
        return OCCPoint(gp_Pnt(self._p.X() + vv.X(), self._p.Y() + vv.Y(), self._p.Z() + vv.Z()))

    def distance_to(self, other: "OCCPoint") -> float:
        require(isinstance(other, OCCPoint), "distance_to expects OCCPoint")
        dx = self._p.X() - other._p.X()
        dy = self._p.Y() - other._p.Y()
        dz = self._p.Z() - other._p.Z()
        return float((dx*dx + dy*dy + dz*dz) ** 0.5)

    def _gp(self) -> gp_Pnt:
        return self._p

    def _serialize(self) -> gp_Pnt:
        return [self._p.Z(), self._p.X(), self._p.Y()]


class OCCVector:
    def __init__(self, *args):
        """
        Accepts:
            OCCVector(gp_Vec)
            OCCVector(x, y, z)
            OCCVector([x, y, z])
        """
        if len(args) == 1 and isinstance(args[0], gp_Vec):
            self._v = args[0]

        elif len(args) == 3:
            x, y, z = args
            self._v = gp_Vec(float(x), float(y), float(z))

        elif len(args) == 1 and isinstance(args[0], (list, tuple)) and len(args[0]) == 3:
            x, y, z = args[0]
            self._v = gp_Vec(float(x), float(y), float(z))

        else:
            raise TypeError(
                "OCCVector expects either (gp_Vec) or (x, y, z) or ([x, y, z]). "
                f"Got args={args}"
            )

    @property
    def x(self) -> float: return float(self._v.X())
    @property
    def y(self) -> float: return float(self._v.Y())
    @property
    def z(self) -> float: return float(self._v.Z())

    def __str__(self):
        return f"[{self._v.X()}, {self._v.Y()}, {self._v.Z()}]"

    def __add__(self, other: "GVector") -> "OCCVector":
        ov = _gp_vec_from_gvec(other)
        return OCCVector(self._v.X() + ov.X(), self._v.Y() + ov.Y(), self._v.Z() + ov.Z())

    def __sub__(self, other: "GVector") -> "OCCVector":
        ov = _gp_vec_from_gvec(other)
        return OCCVector(self._v.X() - ov.X(), self._v.Y() - ov.Y(), self._v.Z() - ov.Z())

    def __mul__(self, s: float) -> "OCCVector":
        s = float(s)
        return OCCVector(self._v.X() * s, self._v.Y() * s, self._v.Z() * s)

    def __rmul__(self, s: float) -> "OCCVector":
        return self.__mul__(s)

    def __neg__(self) -> "OCCVector":
        return OCCVector(-self._v.X(), -self._v.Y(), -self._v.Z())

    def __truediv__(self, s: float) -> "OCCVector":
        s = float(s)
        if s == 0.0:
            raise ZeroDivisionError("Cannot divide vector by zero")
        return OCCVector(self._v.X() / s, self._v.Y() / s, self._v.Z() / s)

    def length(self) -> float:
        return float(self._v.Magnitude())

    def unitized(self) -> "OCCVector":
        v = gp_Vec(self._v.X(), self._v.Y(), self._v.Z())  # copy by components
        if v.Magnitude() == 0.0:
            return OCCVector(v)
        v.Normalize()
        return OCCVector(v)

    def scaled(self, s: float) -> "OCCVector":
        return self.__mul__(s)

    def negated(self) -> "OCCVector":
        return -self

    def rotated(self, angle: float, axis: "OCCVector") -> "OCCVector":
        require(isinstance(axis, OCCVector), "rotated expects axis as OCCVector")
        trsf = gp_Trsf()
        ax1 = gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(axis._v))
        trsf.SetRotation(ax1, float(angle))
        v = gp_Vec(self._v.X(), self._v.Y(), self._v.Z())
        v.Transform(trsf)
        return OCCVector(v)

    def _gp(self) -> gp_Vec:
        return self._v

    def _serialize(self) -> list[float]:
        return [float(self._v.Z()), float(self._v.X()), float(self._v.Y())]


class OCCPlane:
    def __init__(self, *args):
        """
        Accepts:
            OCCPlane(gp_Pln)
            OCCPlane(origin_xyz, normal_xyz)
        """
        if len(args) == 1 and isinstance(args[0], gp_Pln):
            self._pln = args[0]

        elif len(args) == 2:
            origin, normal = args

            ox, oy, oz = origin
            nx, ny, nz = normal

            p = gp_Pnt(float(ox), float(oy), float(oz))
            d = gp_Dir(float(nx), float(ny), float(nz))

            self._pln = gp_Pln(p, d)

        else:
            raise TypeError(
                "OCCPlane expects (gp_Pln) or (origin_xyz, normal_xyz)"
            )

    def z_axis(self) -> "OCCVector":
        n = self._pln.Axis().Direction()
        return OCCVector(gp_Vec(n.X(), n.Y(), n.Z()))

    def origin(self) -> OCCPoint:
        p = self._pln.Location()
        return OCCPoint(p)

    def _gp(self) -> gp_Pln:
        return self._pln

    def _serialize(self):
        p = self._pln.Location()
        n = self._pln.Axis().Direction()

        return {
            "origin": [p.Z(), p.X(), p.Y()],
            "normal": [n.Z(), n.X(), n.Y()],
        }

    def __str__(self):
        s = self._serialize()
        return f"Plane(origin={s['origin']}, normal={s['normal']})"


class OCCLine:
    def __init__(self, *args):
        """
        Accepts:
            OCCLine(gp_Lin)
            OCCLine(a_xyz, b_xyz)
            OCCLine(OCCPoint, OCCPoint)
        Stores a gp_Lin internally, and also caches endpoints if provided.
        """
        self._a: gp_Pnt | None = None
        self._b: gp_Pnt | None = None

        if len(args) == 1 and isinstance(args[0], gp_Lin):
            self._ln = args[0]

        elif len(args) == 2:
            a, b = args

            if isinstance(a, OCCPoint):
                pa = a._gp()
            elif isinstance(a, gp_Pnt):
                pa = a
            else:
                ax, ay, az = a
                pa = gp_Pnt(float(ax), float(ay), float(az))

            if isinstance(b, OCCPoint):
                pb = b._gp()
            elif isinstance(b, gp_Pnt):
                pb = b
            else:
                bx, by, bz = b
                pb = gp_Pnt(float(bx), float(by), float(bz))

            v = gp_Vec(pa, pb)
            require(v.Magnitude() > 0, "Line endpoints cannot be coincident.")
            self._ln = gp_Lin(pa, gp_Dir(v))

            self._a, self._b = pa, pb

        else:
            raise TypeError("OCCLine expects (gp_Lin) or (a, b) endpoints.")

    def from_pt(self) -> OCCPoint:
        return OCCPoint(self._a if self._a is not None else self._ln.Location())

    def to_pt(self) -> OCCPoint:
        if self._b is not None:
            return OCCPoint(self._b)

        # fallback: 1 unit along direction
        p = self._ln.Location()
        d = self._ln.Direction()
        return OCCPoint(gp_Pnt(p.X() + d.X(), p.Y() + d.Y(), p.Z() + d.Z()))
    
    def length(self) -> float:
        return self.from_pt().distance_to(self.to_pt())

    def direction(self) -> OCCVector:
        d = self._ln.Direction()
        return OCCVector(gp_Vec(d.X(), d.Y(), d.Z()))

    def point_at(self, t: float) -> OCCPoint:
        """
        If endpoints are present and t ∈ [0,1], treat t as normalized segment parameter.
        Otherwise treat t as raw infinite-line parameter.
        """
        t = float(t)

        # If this line was constructed from explicit endpoints,
        # and parameter looks normalized, convert to raw infinite parameter
        if self._a is not None and self._b is not None:
            A: gp_Pnt = self._a
            B: gp_Pnt = self._b

            L = A.Distance(B)
            if L > 0:
                # If parameter appears normalized (typical use case)
                # if 0.0 <= t <= 1.0:
                t = t * L

        p = self._ln.Location()
        d = self._ln.Direction()

        return OCCPoint(
            gp_Pnt(
                p.X() + t * d.X(),
                p.Y() + t * d.Y(),
                p.Z() + t * d.Z(),
            )
        )

    def closest_parameter(self, pt: GPoint) -> float:
        p = _gp_pnt_from_gpoint(pt)
        loc = self._ln.Location()
        d = self._ln.Direction()
        vx = p.X() - loc.X()
        vy = p.Y() - loc.Y()
        vz = p.Z() - loc.Z()
        t = vx*d.X() + vy*d.Y() + vz*d.Z()
        return float(t)

    def translated(self, v: GVector) -> "OCCLine":
        vv = _gp_vec_from_gvec(v)

        # translate cached endpoints if present
        if self._a is not None and self._b is not None:
            a2 = gp_Pnt(self._a.X() + vv.X(), self._a.Y() + vv.Y(), self._a.Z() + vv.Z())
            b2 = gp_Pnt(self._b.X() + vv.X(), self._b.Y() + vv.Y(), self._b.Z() + vv.Z())
            return OCCLine(a2, b2)

        # otherwise translate location
        loc = self._ln.Location()
        d = self._ln.Direction()
        new_loc = gp_Pnt(loc.X() + vv.X(), loc.Y() + vv.Y(), loc.Z() + vv.Z())
        return OCCLine(gp_Lin(new_loc, d))

    def _gp(self) -> gp_Lin:
        return self._ln

    def _serialize(self) -> list[list[float]]:
        """
        Serialize as two endpoints: [[ax,ay,az],[bx,by,bz]]
        If endpoints weren't provided, use location + 1 unit along direction.
        """
        a = self.from_pt()._serialize()
        b = self.to_pt()._serialize()
        return [a, b]


class OCCCurve:
    """
    For parity with your RhinoCurve, this adapter treats a curve as a TopoDS_Wire.

    Notes:
    - to_points() is exact for polylines created by this backend (we store points).
    - for arbitrary wires, we return a lightweight vertex list (unique wire vertices).
    """
    def __init__(self, wire: TopoDS_Wire, closed: bool = True, pts_hint: Optional[List[gp_Pnt]] = None, tol: float = 1e-6):
        self._wire = wire
        self._pts_hint = pts_hint  # if created from points
        self._tol = float(tol)
        self._closed = closed

    def duplicate(self) -> "OCCCurve":
        # OCC shapes are immutable-ish; just return a wrapper
        return OCCCurve(self._wire, pts_hint=self._pts_hint, tol=self._tol)

    def translated(self, v: GVector) -> "OCCCurve":

        vv = _gp_vec_from_gvec(v)

        trsf = gp_Trsf()
        trsf.SetTranslation(vv)

        xform = BRepBuilderAPI_Transform(self._wire, trsf, True)
        require(xform.IsDone(), "Curve translation transform failed.")

        shp = xform.Shape()
        require(not shp.IsNull(), "Translated shape is null.")

        # Preserve exact polyline points if available
        pts_hint2 = None
        if self._pts_hint:
            pts_hint2 = [
                gp_Pnt(p.X() + vv.X(), p.Y() + vv.Y(), p.Z() + vv.Z())
                for p in self._pts_hint
            ]

        return OCCCurve(shp, closed=self._closed, pts_hint=pts_hint2, tol=self._tol)

    def to_points(self) -> List[OCCPoint]:
        # Prefer exact points if we built it as a polyline
        if self._pts_hint:
            return [OCCPoint(gp_Pnt(p.X(), p.Y(), p.Z())) for p in self._pts_hint]

        # Otherwise: collect unique vertices from the wire
        pts: List[gp_Pnt] = []
        exp = TopExp_Explorer(self._wire, TopAbs_VERTEX)
        while exp.More():
            v = TopoDS_Vertex(exp.Current())
            p = BRep_Tool.Pnt(v)
            if not any(p.Distance(q) <= self._tol for q in pts):
                pts.append(p)
            exp.Next()
        if self._closed:
            pts += [pts[0]]
        return [OCCPoint(p) for p in pts]

    def contains(self, pt: GPoint, plane: GPlane) -> str:
        """
        Classify point relative to a closed planar polygonal OCCCurve (wire) using ray casting.
        Uses ONLY OCCBackend.curve_line_intersection().

        Assumptions:
        - self.to_points() returns the polyline vertices (and includes closure if self._closed).
        - curve_line_intersection() returns line parameters in the line's [0,1] domain
            (your implementation filters by that already).
        """

        # Convert inputs
        p = _gp_pnt_from_gpoint(pt)
        pln = _gp_pln_from_gplane(plane)

        # Need a backend instance to call the existing intersection routine.
        # Using self._tol keeps behavior consistent with this curve.
        backend = OCCBackend(doc_tol=self._tol)

        # Get plane basis directions (u/v axes in plane)
        ax3: gp_Ax3 = pln.Position()
        xdir = ax3.XDirection()  # gp_Dir
        ydir = ax3.YDirection()  # gp_Dir

        # Jitter the ray direction slightly to reduce "hits exactly at a vertex" double-counting.
        eps = 1e-9
        dx = float(xdir.X()) + eps * float(ydir.X())
        dy = float(xdir.Y()) + eps * float(ydir.Y())
        dz = float(xdir.Z()) + eps * float(ydir.Z())
        dvec = gp_Vec(dx, dy, dz)
        require(dvec.Magnitude() > 0.0, "Degenerate ray direction.")
        dvec.Normalize()

        # Choose a ray length large enough to extend beyond the polygon.
        # We'll use extents of the curve points in the plane coordinate frame.
        pts = self.to_points()
        require(len(pts) >= 2, "contains() requires at least 2 curve points.")

        # Project points to plane UV to get a reasonable scale
        u0, v0 = _plane_point_to_uv(pln, p)
        umin = umax = u0
        vmin = vmax = v0
        for pp in pts:
            gp_pt = pp._gp() if isinstance(pp, OCCPoint) else _gp_pnt_from_gpoint(pp)
            u, v = _plane_point_to_uv(pln, gp_pt)
            umin = min(umin, u); umax = max(umax, u)
            vmin = min(vmin, v); vmax = max(vmax, v)

        span = (umax - umin) + (vmax - vmin)
        R = max(1.0, 10.0 * span + 1.0)  # generous

        # Build ray line segment: from p to p + d*R
        p_far = gp_Pnt(p.X() + dvec.X() * R, p.Y() + dvec.Y() * R, p.Z() + dvec.Z() * R)
        ray = OCCLine(p, p_far)  # endpoints cached, so params are [0,1]

        # Intersect ray with polygon edges (existing method)
        params = backend.curve_line_intersection(self, ray, tol=self._tol)

        if not params:
            return "outside"

        # If we intersect at the ray start, treat as boundary (point on curve)
        # (Your curve_line_intersection can report ~0 when point lies on an edge/vertex.)
        for t in params:
            if abs(float(t)) <= self._tol:
                return "boundary"

        # Count forward intersections (ignore anything behind or extremely near start)
        forward = [float(t) for t in params if float(t) > self._tol and float(t) <= 1.0 + self._tol]

        # Odd/even rule
        return "inside" if (len(forward) % 2 == 1) else "outside"

    def _topods(self) -> TopoDS_Wire:
        return self._wire
    
    def _serialize(self) -> List[List[float]]:
        """
        Serialize curve as a list of [x,y,z] points.
        Prefers exact pts_hint if available; otherwise uses unique vertices.
        """
        pts = self.to_points()
        return [[float(p.z), float(p.x), float(p.y)] for p in pts]


class OCCCircle:
    """
    OCC "circle" can be represented as an edge/wire generated from gp_Circ on a plane.
    This adapter stores the resulting wire.
    """
    def __init__(self, wire: TopoDS_Wire, r: float):
        self._wire = wire
        self._r = float(r)

    def get_curve(self) -> OCCCurve:
        return OCCCurve(self._wire)

    def _topods(self) -> TopoDS_Wire:
        return self._wire


class OCCSolid:
    def __init__(self, shape: TopoDS_Shape):
        self._shape = shape

    def translated(self, v: GVector) -> "OCCSolid":
        vv = _gp_vec_from_gvec(v)
        trsf = gp_Trsf()
        trsf.SetTranslation(vv)
        xform = BRepBuilderAPI_Transform(self._shape, trsf, True)
        require(xform.IsDone(), "Solid translation transform failed.")
        return OCCSolid(xform.Shape())

    def volume(self) -> float:
        props = GProp_GProps()
        brepgprop_VolumeProperties(self._shape, props)
        return float(props.Mass())

    def centroid(self) -> OCCPoint:
        props = GProp_GProps()
        brepgprop_VolumeProperties(self._shape, props)
        c = props.CentreOfMass()
        return OCCPoint(c)

    def bounding_box(self, plane: GPlane) -> Any:
        """
        Returns a plane-aligned bounding box in the plane's coordinate frame.

        The shape is transformed so that:
        - plane origin becomes (0,0,0)
        - plane XDirection becomes +X
        - plane YDirection becomes +Y
        - plane normal (ZDirection) becomes +Z   (i.e. aligned to world Z)

        Output is in your preferred [Z, X, Y] ordering:
        { "min": (zmin, xmin, ymin), "max": (zmax, xmax, ymax) }
        """
        pln = _gp_pln_from_gplane(plane)

        # Plane coordinate system (origin + orthonormal axes)
        plane_ax3 = pln.Position()  # gp_Ax3: Location, XDirection, Direction(Z)

        # World coordinate system
        world_ax3 = gp_Ax3(
            gp_Pnt(0.0, 0.0, 0.0),
            gp_Dir(0.0, 0.0, 1.0),  # world Z
            gp_Dir(1.0, 0.0, 0.0),  # world X
        )

        # Build transform: PlaneCS -> WorldCS, then invert to get World -> Plane
        trsf = gp_Trsf()
        trsf.SetTransformation(plane_ax3, world_ax3)  # map plane axes onto world axes
        trsf.Invert()  # now maps world coords into plane-local coords

        xform = BRepBuilderAPI_Transform(self._shape, trsf, True)
        require(xform.IsDone(), "BBox transform to plane coords failed.")
        shp2 = xform.Shape()

        b = Bnd_Box()
        brepbndlib_Add(shp2, b)
        xmin, ymin, zmin, xmax, ymax, zmax = b.Get()

        # Return in Z, X, Y order (per your convention)
        return {
            "min": (float(xmin), float(ymin), float(zmin)),
            "max": (float(xmax), float(ymax), float(zmax)),
        }

    def _topods(self) -> TopoDS_Shape:
        return self._shape

    def _serialize(self, linear_deflection: float = 0.5, angular_deflection: float = 0.5) -> dict:
        """
        Serialize solid to a triangle mesh payload (JSON/pickle-safe):
          {"v": [[x,y,z],...], "f": [[i,j,k],...]}
        """
        return shape_to_mesh_payload(self._shape, linear_deflection, angular_deflection)


# ======================================================================
# Backend (adapter)
# ======================================================================

class OCCBackend:
    def __init__(self, doc_tol: float | None = TOL):
        self.doc_tol = float(doc_tol if doc_tol is not None else 1e-6)

    # -------------------------
    # CREATE GEO
    # -------------------------

    def point(self, x: float, y: float, z: float) -> OCCPoint:
        return OCCPoint(gp_Pnt(float(x), float(y), float(z)))

    def vector(self, x: float, y: float, z: float) -> OCCVector:
        return OCCVector(gp_Vec(float(x), float(y), float(z)))

    def plane(self, origin: GPoint, normal: GVector) -> OCCPlane:
        o = _gp_pnt_from_gpoint(origin)
        n = _gp_vec_from_gvec(normal)
        require(n.Magnitude() > 0, "Plane normal cannot be zero-length.")
        return OCCPlane(gp_Pln(o, gp_Dir(n)))

    def line(self, a: GPoint, b: GPoint) -> OCCLine:
        pa = _gp_pnt_from_gpoint(a)
        pb = _gp_pnt_from_gpoint(b)
        return OCCLine(pa, pb)

    def curve(self, pts: List[GPoint]) -> OCCCurve:
        require(len(pts) >= 2, "curve requires at least 2 points.")
        gp_pts = [_gp_pnt_from_gpoint(p) for p in pts]

        mk_wire = BRepBuilderAPI_MakeWire()
        for i in range(len(gp_pts) - 1):
            e = BRepBuilderAPI_MakeEdge(gp_pts[i], gp_pts[i + 1]).Edge()
            mk_wire.Add(e)

        wire = mk_wire.Wire()
        require(not wire.IsNull(), "Failed to build wire for curve().")
        return OCCCurve(wire, pts_hint=gp_pts, tol=self.doc_tol)

    def circle(self, plane: GPlane, center: GPoint, r: float) -> OCCCircle:
        """
        Build a circle as a wire (single edge) in the given plane.
        """
        pln = _gp_pln_from_gplane(plane)
        c = _gp_pnt_from_gpoint(center)
        require(float(r) > 0, "Circle radius must be positive.")

        ax2 = gp_Ax2(c, pln.Axis().Direction())  # normal direction
        circ = gp_Circ(ax2, float(r))

        mk = GC_MakeCircle(circ)
        require(mk.IsDone(), "GC_MakeCircle failed.")

        geom = mk.Value()  # Geom_Circle (as Geom_Curve)
        edge = BRepBuilderAPI_MakeEdge(geom).Edge()
        wire = BRepBuilderAPI_MakeWire(edge).Wire()
        return OCCCircle(wire, r=float(r))

    def solid(self, solid: Any) -> OCCSolid:
        require(isinstance(solid, TopoDS_Shape), "solid() expects a TopoDS_Shape")
        return OCCSolid(solid)

    # -------------------------
    # GEO OPS (not owned by classes)
    # -------------------------

    def surface_plane(self, srf: Any) -> OCCPlane:
        """
        Expects a TopoDS_Face.
        """
        require(isinstance(srf, TopoDS_Face), "surface_plane expects TopoDS_Face.")
        adap = BRepAdaptor_Surface(srf, True)
        require(adap.GetType() == GeomAbs_Plane, "Surface is not planar.")
        pln = adap.Plane()
        return OCCPlane(pln)

    def surface_boundary(self, srf: Any) -> OCCCurve:
        """
        Return the outer boundary wire of a planar face as OCCCurve.
        """
        require(isinstance(srf, TopoDS_Face), "surface_boundary expects TopoDS_Face.")
        w = breptools_OuterWire(srf)
        require(not w.IsNull(), "Face has no outer wire.")
        return OCCCurve(w, tol=self.doc_tol) # create Curve from outer wire (no points)

    def surface_boundary_points(self, srf: Any) -> List[OCCPoint]:
        """
        Unique vertices of the outer wire.
        """
        crv = self.surface_boundary(srf)
        return crv.to_points()
    
    def normalize_vector_pair(self, v1: OCCVector, v2: OCCVector):
        dot = (
            v1.x * v2.x +
            v1.y * v2.y +
            v1.z * v2.z
        )

        if dot > 0:
            v1 = v1.negated()
            v2 = v2.negated()

        return (v1, v2)

    # -------------------------
    # INTERSECTIONS
    # -------------------------

    def plane_plane_intersection(self, p1: GPlane, p2: GPlane) -> OCCLine:
        """
        Compute intersection line of two (non-parallel, non-coincident) planes
        using algebra.

        Direction is canonicalized using canonicalize_dir_world()
        so Rhino and OCC always agree.
        """

        pl1 = _gp_pln_from_gplane(p1)
        pl2 = _gp_pln_from_gplane(p2)

        o1 = pl1.Location()
        o2 = pl2.Location()
        n1 = pl1.Axis().Direction()
        n2 = pl2.Axis().Direction()

        # Normals as tuples
        n1v = (float(n1.X()), float(n1.Y()), float(n1.Z()))
        n2v = (float(n2.X()), float(n2.Y()), float(n2.Z()))

        # Plane equation constants
        d1 = _vec_dot(n1v, (float(o1.X()), float(o1.Y()), float(o1.Z())))
        d2 = _vec_dot(n2v, (float(o2.X()), float(o2.Y()), float(o2.Z())))

        # Direction = cross(n1, n2)
        dirv = _vec_cross(n1v, n2v)
        denom = _vec_len2(dirv)

        require(denom > 1e-18,
                "Planes are parallel or coincident; no unique intersection line.")

        # -------------------------------------------------------
        # Canonicalize direction (deterministic sign)
        # -------------------------------------------------------
        dx, dy, dz = canonicalize_dir_world(*dirv, shift=True)

        # -------------------------------------------------------
        # Compute point on line:
        # p = ((d1*n2 - d2*n1) x (n1 x n2)) / |n1 x n2|^2
        # -------------------------------------------------------
        a = (
            d1 * n2v[0] - d2 * n1v[0],
            d1 * n2v[1] - d2 * n1v[1],
            d1 * n2v[2] - d2 * n1v[2],
        )

        pxyz = _vec_cross(a, dirv)

        px = pxyz[0] / denom
        py = pxyz[1] / denom
        pz = pxyz[2] / denom

        p_start = gp_Pnt(px, py, pz)

        # Build second point 1 unit along canonical direction
        d = gp_Dir(dx, dy, dz)
        p_end = gp_Pnt(
            p_start.X() + d.X(),
            p_start.Y() + d.Y(),
            p_start.Z() + d.Z(),
        )

        return OCCLine(p_start, p_end)

    def line_line_params(self, l1: GLine, l2: GLine) -> Tuple[bool, float, float]:
        """
        Closest-approach parameters for two infinite lines.
        Returns (success, a, b) such that P1 = L1(a), P2 = L2(b).

        If endpoints are available on OCCLine (segment), parameters are normalized to [0,1].
        """
        # Keep original objects so we can inspect cached endpoints
        l1_obj = l1 if isinstance(l1, OCCLine) else None
        l2_obj = l2 if isinstance(l2, OCCLine) else None

        L1 = l1._gp() if isinstance(l1, OCCLine) else l1
        L2 = l2._gp() if isinstance(l2, OCCLine) else l2
        require(isinstance(L1, gp_Lin) and isinstance(L2, gp_Lin), "line_line_params expects gp_Lin or OCCLine.")

        p1 = L1.Location()
        p2 = L2.Location()
        u = L1.Direction()
        v = L2.Direction()

        uvec = (float(u.X()), float(u.Y()), float(u.Z()))
        vvec = (float(v.X()), float(v.Y()), float(v.Z()))
        w0 = (float(p1.X() - p2.X()), float(p1.Y() - p2.Y()), float(p1.Z() - p2.Z()))

        a = _vec_dot(uvec, uvec)
        b = _vec_dot(uvec, vvec)
        c = _vec_dot(vvec, vvec)
        d = _vec_dot(uvec, w0)
        e = _vec_dot(vvec, w0)

        denom = a * c - b * b
        if abs(denom) < 1e-18:
            return (False, 0.0, 0.0)

        s = (b * e - c * d) / denom
        t = (a * e - b * d) / denom

        # ---- normalize to [0,1] if endpoints exist ----
        def _normalize_param(line_obj: Optional["OCCLine"], raw_param: float) -> float:
            # Only normalize when we have explicit segment endpoints
            if line_obj is None or getattr(line_obj, "_a", None) is None or getattr(line_obj, "_b", None) is None:
                return float(raw_param)

            A: gp_Pnt = line_obj._a
            B: gp_Pnt = line_obj._b
            L = A.Distance(B)
            if L <= self.doc_tol:
                return 0.0

            u01 = raw_param / float(L)

            # clamp to segment domain
            # if u01 < 0.0: u01 = 0.0
            # if u01 > 1.0: u01 = 1.0
            return float(u01)

        s_norm = _normalize_param(l1_obj, float(s))
        t_norm = _normalize_param(l2_obj, float(t))

        return (True, s_norm, t_norm)

    def curve_line_intersection(self, crv: GCurve, ln: GLine, tol: float = TOL) -> List[float]:
        """
        Intersect a wire-curve with an infinite line using IntCurvesFace_ShapeIntersector.
        Returns line parameters (like Rhino's ParameterB).

        NOTE: Intersector works with shapes; we pass the wire as shape.
        """

        pts = crv.to_points()
        require(len(pts) >= 2, "Polygon curve has too few points.")

        params = []

        for i, a in enumerate(pts[:-1]):
            edge = self.line(a, pts[i + 1])
            success, p_line, p_edge = self.line_line_params(ln, edge)
            if not success:
                continue
            # only add intersections within physical lines [0, 1] domain
            if (-tol <= p_edge <= 1.0 + tol) and (-tol <= p_line <= 1.0 + tol):
                params.append(p_line)

        final_params = unique_floats(params, tol)

        # require(len(final_params) >= 2, "Failed to compute usable intersection params.")

        return final_params

    # -------------------------
    # SOLID OPS
    # -------------------------

    def extrusion(self, profile: GCurve, dir: GVector, amount: float) -> OCCSolid:
        """
        Extrude a planar closed wire along +Z by height.
        If you need extrusion along a plane normal, add a vector argument and use that instead.
        """
        wire = profile._topods() if isinstance(profile, OCCCurve) else profile
        require(isinstance(wire, TopoDS_Wire), "extrusion expects OCCCurve/TopoDS_Wire")

        n = _gp_vec_from_gvec(dir.unitized().scaled(amount))

        # Build a face from the wire (requires planarity)
        face_mk = BRepBuilderAPI_MakeFace(wire, True)
        require(face_mk.IsDone(), "MakeFace(profile) failed; wire may be open or non-planar.")
        face = face_mk.Face()

        prism = BRepPrimAPI_MakePrism(face, n, True)
        require(prism.IsDone(), "MakePrism failed.")
        return OCCSolid(prism.Shape())

    def boolean_difference(self, a: GSolid, b: GSolid) -> OCCSolid:
        sa = a._topods() if isinstance(a, OCCSolid) else a
        sb = b._topods() if isinstance(b, OCCSolid) else b
        require(isinstance(sa, TopoDS_Shape) and isinstance(sb, TopoDS_Shape), "boolean_difference expects shapes.")
        cut = BRepAlgoAPI_Cut(sa, sb)
        cut.SetFuzzyValue(self.doc_tol)
        cut.Build()
        require(cut.IsDone(), "Boolean cut failed.")
        return OCCSolid(cut.Shape())

    def boolean_union(self, solids: List[GSolid]) -> OCCSolid:
        require(len(solids) >= 1, "boolean_union requires at least one solid.")
        shapes = [(s._topods() if isinstance(s, OCCSolid) else s) for s in solids]
        require(all(isinstance(s, TopoDS_Shape) for s in shapes), "boolean_union expects shapes.")

        fused = shapes[0]
        for s in shapes[1:]:
            op = BRepAlgoAPI_Fuse(fused, s)
            op.SetFuzzyValue(self.doc_tol)
            op.Build()
            require(op.IsDone(), "Boolean fuse failed.")
            fused = op.Shape()

        return OCCSolid(fused)