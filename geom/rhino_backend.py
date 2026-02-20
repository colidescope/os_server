# geom/rhino_backend.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple

from geom.protocol import GPoint, GVector, GPlane, GLine, GCurve, GCircle, GSolid
from geom.common import require, canonicalize_dir_world

import Rhino
import Rhino.Geometry as rg

# ======================================================================
# Rhino backend implementation (adapter)
# ======================================================================

class RhinoPoint:
    def __init__(self, *args):
        """
        Accepts:
            RhinoPoint(rg.Point3d)
            RhinoPoint(x, y, z)
        """
        if len(args) == 1 and isinstance(args[0], rg.Point3d):
            self._p = args[0]

        elif len(args) == 3:
            x, y, z = args
            self._p = rg.Point3d(float(x), float(y), float(z))

        else:
            raise TypeError(
                "RhinoPoint expects either (rg.Point3d) or (x, y, z). "
                f"Got args={args}"
            )

    @property
    def x(self): return float(self._p.X)
    @property
    def y(self): return float(self._p.Y)
    @property
    def z(self): return float(self._p.Z)

    def __str__(self):
        return f"[{self._p.X}, {self._p.Y}, {self._p.Z}]"

    def translated(self, v: "GVector") -> "RhinoPoint":
        rv = v._rg() if isinstance(v, RhinoVector) else rg.Vector3d(v.x, v.y, v.z)
        p = rg.Point3d(self._p)
        p.Transform(rg.Transform.Translation(rv))
        return RhinoPoint(p)
    
    def distance_to(self, other: RhinoPoint) -> float:
        return float(self._p.DistanceTo(other._p))

    def _rg(self) -> rg.Point3d:
        return self._p

    def _serialize(self) -> list[float]:
        return [float(self._p.X), float(self._p.Y), float(self._p.Z)]


class RhinoVector:
    def __init__(self, *args):
        """
        Accepts:
            RhinoVector(rg.Vector3d)
            RhinoVector(x, y, z)
            RhinoVector([x, y, z])
        """
        if len(args) == 1 and isinstance(args[0], rg.Vector3d):
            self._v = args[0]

        elif len(args) == 3:
            x, y, z = args
            self._v = rg.Vector3d(float(x), float(y), float(z))

        elif len(args) == 1 and isinstance(args[0], (list, tuple)) and len(args[0]) == 3:
            x, y, z = args[0]
            self._v = rg.Vector3d(float(x), float(y), float(z))

        else:
            raise TypeError(
                "RhinoVector expects either (rg.Vector3d) or (x, y, z) or ([x, y, z]). "
                f"Got args={args}"
            )

    @property
    def x(self): return float(self._v.X)
    @property
    def y(self): return float(self._v.Y)
    @property
    def z(self): return float(self._v.Z)

    def __str__(self):
        return f"[{self._v.X}, {self._v.Y}, {self._v.Z}]"

    def __add__(self, other: "RhinoVector") -> "RhinoVector":
        if isinstance(other, RhinoVector):
            return RhinoVector(self._v + other._v)
        return RhinoVector(self._v + rg.Vector3d(other.x, other.y, other.z))

    def __sub__(self, other: "RhinoVector") -> "RhinoVector":
        if isinstance(other, RhinoVector):
            return RhinoVector(self._v - other._v)
        return RhinoVector(self._v - rg.Vector3d(other.x, other.y, other.z))

    def __mul__(self, s: float) -> "RhinoVector":
        vv = rg.Vector3d(self._v)
        vv *= float(s)
        return RhinoVector(vv)

    def __rmul__(self, s: float) -> "RhinoVector":
        return self.__mul__(s)

    def __neg__(self) -> "RhinoVector":
        vv = rg.Vector3d(self._v)
        vv *= -1.0
        return RhinoVector(vv)

    def __truediv__(self, s: float) -> "RhinoVector":
        if s == 0:
            raise ZeroDivisionError("Cannot divide vector by zero")
        vv = rg.Vector3d(self._v)
        vv /= float(s)
        return RhinoVector(vv)

    def length(self) -> float:
        return float(self._v.Length)

    def unitized(self) -> "RhinoVector":
        vv = rg.Vector3d(self._v)
        vv.Unitize()
        return RhinoVector(vv)

    def scaled(self, s: float) -> "RhinoVector":
        vv = rg.Vector3d(self._v)
        vv *= float(s)
        return RhinoVector(vv)

    def negated(self) -> "RhinoVector":
        return -self

    def rotated(self, angle, axis):
        vv = rg.Vector3d(self._v)
        aa = axis._v if isinstance(axis, RhinoVector) else rg.Vector3d(axis.x, axis.y, axis.z)
        vv.Rotate(float(angle), aa)
        return RhinoVector(vv)

    def _rg(self) -> rg.Vector3d:
        return self._v

    def _serialize(self) -> list[float]:
        return [float(self._v.X), float(self._v.Y), float(self._v.Z)]


class RhinoPlane:
    def __init__(self, *args):
        """
        Accepts:
            RhinoPlane(rg.Plane)
            RhinoPlane(origin_xyz, normal_xyz)
        """
        if len(args) == 1 and isinstance(args[0], rg.Plane):
            self._pl = args[0]

        elif len(args) == 2:
            origin, normal = args

            ox, oy, oz = origin
            nx, ny, nz = normal

            p = rg.Point3d(float(ox), float(oy), float(oz))
            n = rg.Vector3d(float(nx), float(ny), float(nz))

            self._pl = rg.Plane(p, n)

        else:
            raise TypeError(
                "RhinoPlane expects (rg.Plane) or (origin_xyz, normal_xyz)"
            )

    def z_axis(self) -> "RhinoVector":
        return RhinoVector(self._pl.ZAxis)

    def origin(self) -> RhinoPoint:
        return RhinoPoint(self._pl.Origin)

    def _rg(self) -> rg.Plane:
        return self._pl

    def _serialize(self):
        return {
            "origin": [self._pl.Origin.X, self._pl.Origin.Y, self._pl.Origin.Z],
            "normal": [self._pl.ZAxis.X, self._pl.ZAxis.Y, self._pl.ZAxis.Z],
        }

    def __str__(self):
        s = self._serialize()
        return f"Plane(origin={s['origin']}, normal={s['normal']})"


class RhinoLine:
    def __init__(self, *args):
        """
        Accepts:
            RhinoLine(rg.Line)
            RhinoLine(a_xyz, b_xyz)
            RhinoLine(RhinoPoint, RhinoPoint)
        """
        if len(args) == 1 and isinstance(args[0], rg.Line):
            self._ln = args[0]

        elif len(args) == 2:
            a, b = args

            if isinstance(a, RhinoPoint):
                pa = a._rg()
            elif isinstance(a, rg.Point3d):
                pa = a
            else:
                ax, ay, az = a
                pa = rg.Point3d(float(ax), float(ay), float(az))

            if isinstance(b, RhinoPoint):
                pb = b._rg()
            elif isinstance(b, rg.Point3d):
                pb = b
            else:
                bx, by, bz = b
                pb = rg.Point3d(float(bx), float(by), float(bz))

            self._ln = rg.Line(pa, pb)

        else:
            raise TypeError("RhinoLine expects (rg.Line) or (a, b) endpoints.")

    def from_pt(self) -> RhinoPoint:
        return RhinoPoint(self._ln.From)

    def to_pt(self) -> RhinoPoint:
        return RhinoPoint(self._ln.To)

    def direction(self) -> RhinoVector:
        return RhinoVector(self._ln.Direction)

    def point_at(self, t: float) -> RhinoPoint:
        return RhinoPoint(self._ln.PointAt(t))

    def closest_parameter(self, pt: GPoint) -> float:
        rp = pt._rg() if isinstance(pt, RhinoPoint) else rg.Point3d(pt.x, pt.y, pt.z)
        return float(self._ln.ClosestParameter(rp))

    def translated(self, v: GVector) -> "RhinoLine":
        rv = v._rg() if isinstance(v, RhinoVector) else rg.Vector3d(v.x, v.y, v.z)
        ln = rg.Line(self._ln.From, self._ln.To)
        ln.Transform(rg.Transform.Translation(rv))
        return RhinoLine(ln)

    def _rg(self) -> rg.Line:
        return self._ln

    def _serialize(self) -> list[list[float]]:
        a = [self._ln.From.X, self._ln.From.Y, self._ln.From.Z]
        b = [self._ln.To.X, self._ln.To.Y, self._ln.To.Z]
        return [a, b]


class RhinoCurve:
    def __init__(self, *args):
        """
        Accepts:
            RhinoCurve(rg.Curve)
            RhinoCurve([[x,y,z], ...])                # polyline-like
            RhinoCurve([RhinoPoint|rg.Point3d|...])   # optional flexibility
        """
        self.doc_tol = float(Rhino.RhinoDoc.ActiveDoc.ModelAbsoluteTolerance)

        if len(args) == 1 and isinstance(args[0], rg.Curve):
            self._crv = args[0]

        elif len(args) == 1 and isinstance(args[0], (list, tuple)):
            pts_in = args[0]
            if len(pts_in) < 2:
                raise ValueError("RhinoCurve(points) requires at least 2 points.")

            # Normalize inputs to rg.Point3d
            pts = []
            for item in pts_in:
                if isinstance(item, RhinoPoint):
                    pts.append(item._rg())
                elif isinstance(item, rg.Point3d):
                    pts.append(item)
                elif isinstance(item, (list, tuple)) and len(item) == 3:
                    x, y, z = item
                    pts.append(rg.Point3d(float(x), float(y), float(z)))
                else:
                    raise TypeError(f"Unsupported point item in RhinoCurve(points): {type(item)}")

            poly = rg.Polyline(pts)
            require(poly.IsValid, "Invalid polyline from points.")
            self._crv = poly.ToNurbsCurve()

        else:
            raise TypeError(
                "RhinoCurve expects either (rg.Curve) or (points_list). "
                f"Got args={args}"
            )

    def duplicate(self) -> "RhinoCurve":
        return RhinoCurve(self._crv.DuplicateCurve())

    def translated(self, v: GVector) -> "RhinoCurve":
        rv = v._rg() if isinstance(v, RhinoVector) else rg.Vector3d(v.x, v.y, v.z)
        c = self._crv.DuplicateCurve()
        c.Transform(rg.Transform.Translation(rv))
        return RhinoCurve(c)

    def to_points(self) -> List[RhinoPoint]:
        return [RhinoPoint(p) for p in list(self._crv.ToArray())]

    def contains(self, pt: GPoint, plane: GPlane) -> str:
        rp = pt._rg() if isinstance(pt, RhinoPoint) else rg.Point3d(pt.x, pt.y, pt.z)
        rpl = plane._rg() if isinstance(plane, RhinoPlane) else plane
        contains = self._crv.ToNurbsCurve().Contains(rp, rpl, self.doc_tol)
        if contains == rg.PointContainment.Inside:
            return "inside"
        if contains == rg.PointContainment.Outside:
            return "outside"
        return "boundary"

    def _rg(self) -> rg.Curve:
        return self._crv

    def _serialize(self) -> List[List[float]]:
        """
        Serialize curve as a list of [x,y,z] points.
        Uses Rhino's ToArray() which returns a polyline-ish sampling for some curve types.
        """
        pts = list(self._crv.ToArray())
        return [[float(p.X), float(p.Y), float(p.Z)] for p in pts]


class RhinoCircle:
    def __init__(self, c: rg.Circle):
        self._c = c

    def get_curve(self) -> RhinoCurve:
        return RhinoCurve(self._c.ToNurbsCurve().DuplicateCurve())

    def _rg(self) -> rg.Circle:
        return self._c


class RhinoSolid:
    def __init__(self, brep_or_extrusion):
        self._g = brep_or_extrusion  # Brep or Extrusion

    def _brep(self) -> rg.Brep:
        if isinstance(self._g, rg.Brep):
            return self._g
        return self._g.ToBrep()

    def translated(self, v: GVector) -> "RhinoSolid":
        rv = v._rg() if isinstance(v, RhinoVector) else rg.Vector3d(v.x, v.y, v.z)
        brep = self._brep().Duplicate()
        brep.Transform(rg.Transform.Translation(rv))
        return RhinoSolid(brep)

    def volume(self) -> float:
        return float(self._brep().GetVolume())

    def centroid(self) -> RhinoPoint:
        mp = rg.VolumeMassProperties.Compute(self._brep())
        require(mp is not None, "VolumeMassProperties.Compute failed.")
        return RhinoPoint(mp.Centroid)

    def bounding_box(self, plane: GPlane) -> Any:
        rpl = plane._rg() if isinstance(plane, RhinoPlane) else plane
        bb = self._brep().GetBoundingBox(rpl)
        return {
            "min": (float(bb.Min.X), float(bb.Min.Y), float(bb.Min.Z)),
            "max": (float(bb.Max.X), float(bb.Max.Y), float(bb.Max.Z)),
        }

    def _rg(self):
        return self._g


class RhinoMesh:
    def __init__(self, *args):
        """
        Accepts:
          RhinoMesh(rg.Mesh)
          RhinoMesh(mesh_payload) where mesh_payload = {"v": [[x,y,z],...], "f": [[i,j,k],...]}
        """
        if len(args) == 1 and isinstance(args[0], rg.Mesh):
            self._m = args[0]

        elif len(args) == 1 and isinstance(args[0], dict):
            payload = args[0]
            verts = payload.get("v", None)
            faces = payload.get("f", None)
            require(isinstance(verts, list) and isinstance(faces, list), "Invalid mesh payload")

            m = rg.Mesh()

            # Add vertices
            for x, y, z in verts:
                m.Vertices.Add(float(x), float(y), float(z))

            # Add faces (triangles)
            for f in faces:
                require(len(f) == 3, "Only triangle faces supported in payload")
                i, j, k = f
                m.Faces.AddFace(int(i), int(j), int(k))

            m.Normals.ComputeNormals()
            m.Compact()
            self._m = m

        else:
            raise TypeError("RhinoMesh expects (rg.Mesh) or (mesh_payload dict)")

    def _rg(self) -> rg.Mesh:
        return self._m

    def _serialize(self) -> dict:
        """
        Optional: serialize Rhino mesh back to same payload schema.
        """
        verts = [[v.X, v.Y, v.Z] for v in self._m.Vertices]
        faces = []
        for f in self._m.Faces:
            if f.IsTriangle:
                faces.append([f.A, f.B, f.C])
            else:
                # convert quad -> 2 tris
                faces.append([f.A, f.B, f.C])
                faces.append([f.A, f.C, f.D])
        return {"v": verts, "f": faces}


class RhinoBackend:
    def __init__(self, doc_tol: float | None = None):
        self.doc_tol = float(doc_tol if doc_tol is not None else Rhino.RhinoDoc.ActiveDoc.ModelAbsoluteTolerance)

    ### CREATE GEO

    def point(self, x: float, y: float, z: float) -> RhinoPoint:
        return RhinoPoint(rg.Point3d(x, y, z))

    def vector(self, x: float, y: float, z: float) -> RhinoVector:
        return RhinoVector(rg.Vector3d(x, y, z))

    def plane(self, origin: GPoint, normal: GVector) -> RhinoPlane:
        o = (origin._rg())
        n = (normal._rg())
        require(n.Length > 0, "Plane normal cannot be zero-length.")
        return RhinoPlane(rg.Plane(o, n))

    def line(self, a: GPoint, b: GPoint) -> RhinoLine:
        ra = a._rg() if isinstance(a, RhinoPoint) else rg.Point3d(a.x, a.y, a.z)
        rb = b._rg() if isinstance(b, RhinoPoint) else rg.Point3d(b.x, b.y, b.z)
        return RhinoLine(rg.Line(ra, rb))
    
    def curve(self, pts: List[GPoint]) -> RhinoCurve:
        poly = rg.Polyline([pt._rg() for pt in pts])
        return RhinoCurve(poly.ToNurbsCurve())

    def circle(self, plane: GPlane, center: GPoint, r: float) -> RhinoCircle:
        rpl = plane._rg() if isinstance(plane, RhinoPlane) else plane
        rc = center._rg() if isinstance(center, RhinoPoint) else rg.Point3d(center.x, center.y, center.z)
        return RhinoCircle(rg.Circle(rpl, rc, r))
    
    def solid(self, solid: Any) -> RhinoSolid:
        return RhinoSolid(solid)
    
    ### GEO OPERATIONS (NOT OWNED BY GEO CLASSES ABOVE)

    def surface_plane(self, srf: Any) -> RhinoPlane:
        ok, plane = srf.TryGetPlane()
        require(ok, "Surface is not planar (TryGetPlane failed).")
        return RhinoPlane(plane)

    def surface_boundary(self, srf: Any) -> RhinoCurve:
        curves = srf.ToBrep().DuplicateEdgeCurves()
        require(curves and len(curves) > 0, "Surface has no edge curves.")
        joined = rg.Curve.JoinCurves(curves, self.doc_tol)
        require(joined and len(joined) > 0, "Failed to join surface edge curves.")
        require(len(joined) == 1, f"Expected 1 joined boundary curve; got {len(joined)}.")
        return RhinoCurve(joined[0])

    def surface_boundary_points(self, srf: Any) -> List[RhinoPoint]:
        curves = srf.ToBrep().DuplicateEdgeCurves()
        require(curves and len(curves) > 0, "Surface has no edge curves.")
        pts: List[rg.Point3d] = []
        for crv in curves:
            for cand in [crv.PointAtStart, crv.PointAtEnd]:
                if not any(cand.DistanceTo(p) <= self.doc_tol for p in pts):
                    pts.append(cand)
        require(len(pts) >= 2, "Failed to extract boundary points.")
        return [RhinoPoint(p) for p in pts]
    
    def normalize_vector_pair(self, v1: RhinoVector, v2: RhinoVector):
        dot = v1._rg() * v2._rg()  # Rhino operator overload

        if dot < 0:
            v1 = v1.negated()
            v2 = v2.negated()

        return (v1, v2)
        
    ### INTERSECTIONS

    def plane_plane_intersection(self, p1: GPlane, p2: GPlane) -> RhinoLine:
        rp1 = p1._rg() if isinstance(p1, RhinoPlane) else p1
        rp2 = p2._rg() if isinstance(p2, RhinoPlane) else p2
        ok, line = rg.Intersect.Intersection.PlanePlane(rp1, rp2)
        require(ok, "Planes are parallel or coincident; no unique intersection line.")
        
        d = line.Direction
        dx, dy, dz = canonicalize_dir_world(d.X, d.Y, d.Z)

        # rebuild line with canonical direction
        from_pt = line.From
        to_pt = rg.Point3d(from_pt.X + dx, from_pt.Y + dy, from_pt.Z + dz)
        
        return RhinoLine(rg.Line(from_pt, to_pt))
        # return RhinoLine(line)

    def line_line_params(self, l1: GLine, l2: GLine) -> Tuple[bool, float, float]:
        rl1 = l1._rg() if isinstance(l1, RhinoLine) else l1
        rl2 = l2._rg() if isinstance(l2, RhinoLine) else l2
        success, a, b = rg.Intersect.Intersection.LineLine(rl1, rl2)
        return bool(success), float(a), float(b)

    def curve_line_intersection(self, crv: GCurve, ln: GLine) -> List[float]:
        rcrv = crv._rg() if isinstance(crv, RhinoCurve) else crv
        rln = ln._rg() if isinstance(ln, RhinoLine) else ln
        inters = rg.Intersect.Intersection.CurveLine(rcrv, rln, self.doc_tol, self.doc_tol)
        if inters is None or inters.Count == 0:
            return []
        return [float(i.ParameterB) for i in inters]

    # SOLID OPERATIONS

    def extrusion(self, profile: GCurve, dir: GVector, amount: float) -> RhinoSolid:
        rcrv = profile._rg()
        point = rcrv.PointAt(0)
        plane = rg.Plane(point, dir._rg())
        ex = rg.Extrusion.Create(rcrv, plane, amount, True)
        require(ex is not None, "Extrusion.Create failed.")
        return RhinoSolid(ex)

    def boolean_difference(self, a: GSolid, b: GSolid) -> RhinoSolid:
        ra = a._brep() if isinstance(a, RhinoSolid) else a
        rb = b._brep() if isinstance(b, RhinoSolid) else b
        diff = rg.Brep.CreateBooleanDifference(ra, rb, self.doc_tol)
        require(diff and len(diff) > 0, "BooleanDifference failed.")
        return RhinoSolid(diff[0])

    def boolean_union(self, solids: List[GSolid]) -> RhinoSolid:
        breps = [(s._brep() if isinstance(s, RhinoSolid) else s) for s in solids]
        union = rg.Brep.CreateBooleanUnion(breps, self.doc_tol)
        require(union and len(union) > 0, "BooleanUnion failed.")
        return RhinoSolid(union[0])
