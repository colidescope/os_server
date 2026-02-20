# geom/mesh_to_face.py

import os
import tempfile
import base64
from uuid import uuid4

from typing import Dict, List, Tuple

from geom.common import require

# ---------------------------
# pythonocc-core imports
# ---------------------------
from OCC.Core.gp import gp_Pnt

from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_COMPOUND, TopAbs_SHELL, TopAbs_SOLID
from OCC.Core.TopoDS import TopoDS_Shape, topods, TopoDS_Iterator, TopoDS_Face, TopoDS_Solid

from OCC.Core.GProp import GProp_GProps

from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_MakePolygon,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_Sewing,
    BRepBuilderAPI_MakeSolid,
)

import OCC.Core.BRepTools as BRepTools
from OCC.Core.BRep import BRep_Builder, BRep_Tool

from OCC.Core.BRepGProp import brepgprop_SurfaceProperties
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeSphere
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.ShapeFix import ShapeFix_Solid

from OCC.Core.StlAPI import StlAPI_Writer
from OCC.Extend.DataExchange import read_step_file, write_step_file
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IGESControl import IGESControl_Reader

from OCC.Core.TopLoc import TopLoc_Location

### MESH SURFACE TO OCC FACE ###

def _positions_dict_to_vertices(pos: Dict[str, float], item_size: int = 3) -> List[Tuple[float,float,float]]:
    # positions is {"0": x0, "1": y0, ...} in your payload
    arr = [0.0] * len(pos)
    for k, v in pos.items():
        arr[int(k)] = float(v)
    if item_size != 3:
        raise ValueError(f"Expected itemSize=3, got {item_size}")
    if len(arr) % 3 != 0:
        raise ValueError("positions length not divisible by 3")
    verts = []
    for i in range(0, len(arr), 3):
        verts.append((arr[i], arr[i+1], arr[i+2]))
    return verts

def _indices_dict_to_tris(ind: Dict[str, int]) -> List[Tuple[int,int,int]]:
    arr = [0] * len(ind)
    for k, v in ind.items():
        arr[int(k)] = int(v)
    if len(arr) % 3 != 0:
        raise ValueError("indices length not divisible by 3")
    tris = []
    for i in range(0, len(arr), 3):
        tris.append((arr[i], arr[i+1], arr[i+2]))
    return tris

def _edge_key(a: int, b: int) -> Tuple[int,int]:
    return (a,b) if a < b else (b,a)

def _boundary_loop_from_tris(tris: List[Tuple[int,int,int]]) -> List[int]:
    """
    Returns ordered boundary vertex indices for a single outer loop.
    Assumes the input is a single planar surface with a single boundary.
    """
    edge_count = {}
    for (i0,i1,i2) in tris:
        for a,b in [(i0,i1),(i1,i2),(i2,i0)]:
            k = _edge_key(a,b)
            edge_count[k] = edge_count.get(k, 0) + 1

    boundary_edges = [e for e,c in edge_count.items() if c == 1]
    if len(boundary_edges) < 3:
        raise ValueError("Could not find boundary edges (mesh may not be a single open surface).")

    # Build adjacency on boundary (each boundary vertex should connect to 2)
    adj = {}
    for a,b in boundary_edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    # Find a start vertex (any)
    start = boundary_edges[0][0]
    loop = [start]
    prev = None
    cur = start

    # Walk until we return to start
    for _ in range(100000):
        nbrs = adj.get(cur, [])
        if len(nbrs) == 0:
            raise ValueError("Broken boundary adjacency.")
        # choose next != prev
        if prev is None:
            nxt = nbrs[0]
        else:
            nxt = nbrs[0] if nbrs[0] != prev else (nbrs[1] if len(nbrs) > 1 else None)
            if nxt is None:
                raise ValueError("Open boundary (unexpected).")

        if nxt == start:
            break

        loop.append(nxt)
        prev, cur = cur, nxt

        if len(loop) > len(adj) + 5:
            raise ValueError("Boundary walk did not close (bad mesh).")

    if len(loop) < 3:
        raise ValueError("Boundary loop too small.")

    return loop

def mesh_surface_to_occ_face(positions: Dict[str, float], indices: Dict[str, int], item_size: int = 3) -> TopoDS_Face:
    verts = _positions_dict_to_vertices(positions, item_size)
    tris = _indices_dict_to_tris(indices)
    loop_ids = _boundary_loop_from_tris(tris)

    poly = BRepBuilderAPI_MakePolygon()
    for vid in loop_ids:
        x,y,z = verts[vid]
        poly.Add(gp_Pnt(x,y,z))
    poly.Close()

    if not poly.IsDone():
        raise ValueError("Failed to build polygon from boundary loop.")

    face_mk = BRepBuilderAPI_MakeFace(poly.Wire(), True)
    if not face_mk.IsDone():
        raise ValueError("Failed to build planar face from polygon wire.")

    return face_mk.Face()

def face_area(face) -> float:
    props = GProp_GProps()
    brepgprop_SurfaceProperties(face, props)
    return props.Mass()  # for SurfaceProperties, "Mass" == area

### SOLID MESH TO OCC SOLID ###

def mesh_to_occ_solid(
    positions: Dict[str, float],
    indices: Dict[str, int],
    item_size: int = 3,
    *,
    tol: float = 1e-6,
    assume_positions_are_zxy: bool = False,
    fix: bool = True,
) -> TopoDS_Solid:
    """
    Build a TopoDS_Solid from a triangle mesh.

    positions: dict of floats (flat array) in itemSize groups
    indices: dict of ints (flat array) in triples
    assume_positions_are_zxy:
        If True, interpret each vertex triple as [Z, X, Y] and convert to (X,Y,Z).
        If False, interpret as standard [X, Y, Z].
    """

    verts = _positions_dict_to_vertices(positions, item_size)
    tris = _indices_dict_to_tris(indices)

    # Optional unswizzle: [Z,X,Y] -> (X,Y,Z)
    if assume_positions_are_zxy:
        verts = [(x, y, z) for (z, x, y) in verts]

    # Build & sew triangle faces into a shell
    sewing = BRepBuilderAPI_Sewing(tol, True, True, True, False)

    for (ia, ib, ic) in tris:
        ax, ay, az = verts[ia]
        bx, by, bz = verts[ib]
        cx, cy, cz = verts[ic]

        poly = BRepBuilderAPI_MakePolygon()
        poly.Add(gp_Pnt(ax, ay, az))
        poly.Add(gp_Pnt(bx, by, bz))
        poly.Add(gp_Pnt(cx, cy, cz))
        poly.Close()

        if not poly.IsDone():
            raise ValueError("Failed to build triangle polygon")

        face_mk = BRepBuilderAPI_MakeFace(poly.Wire(), True)
        if not face_mk.IsDone():
            raise ValueError("Failed to build triangle face")

        sewing.Add(face_mk.Face())

    sewing.Perform()
    sewed: TopoDS_Shape = sewing.SewedShape()
    if sewed.IsNull():
        raise ValueError("Sewing produced a null shape (mesh may be invalid)")

    # Extract a shell from the sewed shape (works across pythonocc builds without casting constructors)
    shell = None
    exp = TopExp_Explorer(sewed, TopAbs_SHELL)
    if exp.More():
        shell = exp.Current()
    else:
        # Some cases: sewed may itself be a shell-like object; still try MakeSolid directly below
        shell = sewed

    # Make solid from shell
    mk_solid = BRepBuilderAPI_MakeSolid()
    mk_solid.Add(shell)
    if not mk_solid.IsDone():
        raise ValueError("BRepBuilderAPI_MakeSolid failed (mesh likely not closed/watertight)")

    solid = mk_solid.Solid()

    # Optional: repair/fix
    if fix:
        sfix = ShapeFix_Solid(solid)
        sfix.Perform()
        solid = sfix.Solid()

    # Validate
    ana = BRepCheck_Analyzer(solid)
    if not ana.IsValid():
        # Still return it if you want, but volume may fail / be 0 / be nonsense.
        raise ValueError("Resulting solid is not valid (mesh may not be closed or has inconsistencies)")

    return solid

### CONVERT HELPERS ###

def read_shape_with_occ(path: str) -> TopoDS_Shape:
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.step', '.stp']:
        reader = STEPControl_Reader()
        status = reader.ReadFile(path)
        if status != 1:
            raise RuntimeError('Failed to read STEP file')
        reader.TransferRoots()
        return reader.OneShape()

    if ext in ['.iges', '.igs']:
        reader = IGESControl_Reader()
        status = reader.ReadFile(path)
        if status != 1:
            raise RuntimeError('Failed to read IGES file')
        reader.TransferRoots()
        return reader.OneShape()

    if ext == '.brep':
        shape = TopoDS_Shape()
        builder = BRep_Builder()
        if not BRepTools.Read(shape, path, builder):
            raise RuntimeError('Failed to read BREP file')
        return shape

    raise RuntimeError(f'Unsupported extension: {ext}')

def shape_to_tri_mesh(
    shape: TopoDS_Shape,
    lin_deflection: float = 0.5,
    ang_deflection: float = 0.5,
):
    """
    Return (elements, total_vertex_count, total_triangle_count), where elements
    is a list of {"vertices": [...], "faces": [...]}.

    Grouping strategy:
      1) Group faces by owning SOLID (typical 'parts')
      2) If no SOLIDs, group by owning SHELL
      3) If neither, split a COMPOUND by its direct children
      4) Else, put all faces in one element
    """

    # Mesh once; creates per-face triangulations. This does NOT merge parts.
    BRepMesh_IncrementalMesh(shape, lin_deflection, False, ang_deflection, True)

    def triangulate_face_group(faces_group):
        surfaces = []

        for face in faces_group:
            vertices = []
            faces = []
            vmap = {}
            
            loc = TopLoc_Location()
            htri = BRep_Tool.Triangulation(face, loc)
            if not htri:
                continue
            tri = getattr(htri, "GetObject", lambda: htri)()
            if not tri or tri.NbTriangles() == 0:
                continue

            trsf = loc.Transformation()

            # OCC uses 1-based indexing for triangulations
            for ti in range(1, tri.NbTriangles() + 1):
                t = tri.Triangle(ti)
                i1, i2, i3 = t.Get()
                tri_idx = []
                for ii in (i1, i2, i3):
                    p = tri.Node(ii).Transformed(trsf)
                    key = (round(p.X()*1_000_000), round(p.Y()*1_000_000), round(p.Z()*1_000_000))
                    idx = vmap.get(key)
                    if idx is None:
                        idx = len(vertices)
                        vmap[key] = idx
                        vertices.append([float(p.X()), float(p.Y()), float(p.Z())])
                    tri_idx.append(idx)
                faces.append(tri_idx)

            surfaces.append({"vertices": vertices, "faces": faces})

        return surfaces, len(vertices), len(faces)

    def groups_by_owner_explore(root: TopoDS_Shape, owner_type):
        """
        Build groups of faces by iterating each owner (SOLID or SHELL)
        and collecting the faces under that owner. No MapShapesAndAncestors needed.
        """
        groups = []
        owner_exp = TopExp_Explorer(root, owner_type)
        while owner_exp.More():
            owner = owner_exp.Current()
            face_list = []
            face_exp = TopExp_Explorer(owner, TopAbs_FACE)
            while face_exp.More():
                face_list.append(topods.Face(face_exp.Current()))
                face_exp.Next()
            if face_list:
                groups.append(face_list)
            owner_exp.Next()
        return groups

    # 1) Try SOLIDs
    face_groups = groups_by_owner_explore(shape, TopAbs_SOLID)
    print("solid_groups: {}".format(face_groups))

    # 2) Else SHELLs
    if not face_groups:
        face_groups = groups_by_owner_explore(shape, TopAbs_SHELL)
        print("shell_groups: {}".format(face_groups))

    # 3) Else split compound by direct children (each child gets its faces)
    if not face_groups and shape.ShapeType() == TopAbs_COMPOUND:
        parts = []
        it = TopoDS_Iterator(shape)
        while it.More():
            child = it.Value()
            faces_group = []
            face_exp = TopExp_Explorer(child, TopAbs_FACE)
            while face_exp.More():
                faces_group.append(topods.Face(face_exp.Current()))
                face_exp.Next()
            if faces_group:
                parts.append(faces_group)
            it.Next()
        face_groups = parts
        print("child_groups: {}".format(face_groups))

    # 4) Else, one group with all faces (if any)
    if not face_groups:
        all_faces = []
        exp = TopExp_Explorer(shape, TopAbs_FACE)
        while exp.More():
            all_faces.append(topods.Face(exp.Current()))
            exp.Next()
        if all_faces:
            face_groups = [all_faces]
            print("final_groups: {}".format(face_groups))
        else:
            return [], 0, 0

    # Triangulate each group into its own element
    elements = []
    total_v = 0
    total_f = 0

    for faces_group in face_groups:
        elem, nv, nf = triangulate_face_group(faces_group)
        if nv > 0 and nf > 0:
            elements.append(elem)
            total_v += nv
            total_f += nf

    return elements, total_v, total_f


### SERIALIZE OCC SHAPE TO MESH JSON ###

# def trimesh_to_arrays(mesh: 'trimesh.Trimesh') -> Tuple[List[List[float]], List[List[int]]]:
#     v = mesh.vertices.tolist()
#     f = mesh.faces.tolist()
#     return { "vertices": v, "faces": f}, len(v), len(f)

def shape_to_mesh_payload(shape, linear_deflection: float = 0.5, angular_deflection: float = 0.5):
    require(not shape.IsNull(), "shape_to_mesh_payload: shape is null")

    mesher = BRepMesh_IncrementalMesh(shape, float(linear_deflection), False, float(angular_deflection), True)
    mesher.Perform()
    require(mesher.IsDone(), "BRepMesh_IncrementalMesh failed")

    verts: List[List[float]] = []
    faces: List[List[int]] = []
    vmap: Dict[Tuple[int, int, int], int] = {}

    def vkey(z: float, x: float, y: float, scale: float = 1e6):
        return (int(round(z * scale)), int(round(x * scale)), int(round(y * scale)))

    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = exp.Current()

        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation(face, loc)
        if tri is None:
            exp.Next()
            continue

        trsf = loc.Transformation()
        local_to_global: Dict[int, int] = {}

        nn = tri.NbNodes()
        for i in range(1, nn + 1):
            p = tri.Node(i)  # your build supports Node(i)
            p.Transform(trsf)

            x = float(p.X())
            y = float(p.Y())
            z = float(p.Z())

            # 🔁 reorder to Z, X, Y
            k = vkey(z, x, y)
            gi = vmap.get(k)
            if gi is None:
                gi = len(verts)
                vmap[k] = gi
                verts.append([z, x, y])  # ← reordered here

            local_to_global[i] = gi

        nt = tri.NbTriangles()
        for i in range(1, nt + 1):
            t = tri.Triangle(i)
            n1, n2, n3 = t.Get()
            faces.append([
                local_to_global[n1],
                local_to_global[n2],
                local_to_global[n3],
            ])

        exp.Next()

    require(verts and faces, "shape_to_mesh_payload produced empty mesh")

    return {
        "v": verts,  # vertices now [Z, X, Y]
        "f": faces,
    }


### EXPORTERS (NOT USED) ###

def occ_to_step_base64(shape) -> str:
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, f"{uuid4()}.step")
        write_step_file(shape, p)
        with open(p, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

def occ_to_stl_base64(shape, lin_deflection=0.25, ang_deflection=0.25) -> str:
    # mesh
    BRepMesh_IncrementalMesh(shape, lin_deflection, True, ang_deflection, True)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, f"{uuid4()}.stl")
        w = StlAPI_Writer()
        w.Write(shape, p)
        with open(p, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
