# bracket/bracket_script.py
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from helpers import clear_log_file, log
from geom.common import TOL, require, unique_floats
from geom.protocol import (
    GeomBackend,
    GCircle,
    GCurve,
    GLine,
    GPlane,
    GPoint,
    GVector,
    GSolid,
    Interval,
)


def _material_value(
    material: Dict[str, Any],
    *keys: str,
    required: bool = True,
    default: Optional[float] = None,
) -> Optional[float]:
    """
    Returns the first matching numeric material property from a material dict.
    """
    if not material:
        if required:
            raise ValueError("Material data is missing.")
        return default

    for key in keys:
        if key in material and material[key] is not None:
            return float(material[key])

    if required:
        raise KeyError(
            f"Could not find any of the required material keys: {keys}. "
            f"Available keys: {list(material.keys())}"
        )

    return default


def _safe_serialize(value: Any) -> Any:
    """
    Best-effort serializer for logging geometry-ish values and numpy arrays.
    """
    if value is None:
        return None

    if hasattr(value, "_serialize"):
        try:
            return value._serialize()
        except Exception:
            pass

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, dict):
        return {k: _safe_serialize(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_safe_serialize(v) for v in value]

    return value


def _fmt_number(value: Any, digits: int = 6) -> str:
    if isinstance(value, (int, float, np.floating, np.integer)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _fmt_vector(value: Any, digits: int = 6) -> str:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return (
            f"({_fmt_number(value[0], digits)}, "
            f"{_fmt_number(value[1], digits)}, "
            f"{_fmt_number(value[2], digits)})"
        )
    return str(value)


def _fmt_json(value: Any) -> str:
    return json.dumps(_safe_serialize(value), indent=2)


def _pad_lines(text: str, prefix: str = "  ") -> str:
    return "\n".join(prefix + line if line else line for line in text.splitlines())


def main(
    geom: GeomBackend,
    srf_1: Any,
    srf_2: Any,
    max_bolt_dia: float,
    min_edge_clearance: float,
    bolt_spacing_1: float,
    bolt_spacing_2: float,
    init_bolt_radius: float,
    init_plate_thickness: float,
    num_ribs: int,
    material_free: Optional[Dict[str, Any]] = None,
    material_bracket: Optional[Dict[str, Any]] = None,
    free_objects: Optional[List[Dict[str, Any]]] = None,
    tol: float = TOL,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Bracket generator.

    - srf_1, srf_2: backend surface/face objects understood by `geom.surface_*`
    - material_free: full material dict for the free object
    - material_bracket: full material dict for the bracket
    - free_objects: list of dicts like:
        {
            "cg": [x, y, z],
            "volume": ...,
            "material": {...}  # optional
        }
    - returns backend geometry objects depending on the backend
    """

    log_record = ""
    clear_log_file()

    def record(message: Any) -> None:
        nonlocal log_record
        log_record += log(message)

    def record_line(text: str = "") -> None:
        record(text + "\n")

    def record_section(title: str) -> None:
        record("\n" + "=" * 72 + "\n")
        record(f"{title}\n")
        record("=" * 72 + "\n")

    def record_subsection(title: str) -> None:
        record("\n" + "-" * 48 + "\n")
        record(f"{title}\n")
        record("-" * 48 + "\n")

    def record_kv(items: Dict[str, Any], digits: int = 6) -> None:
        if not items:
            return
        width = max(len(str(k)) for k in items.keys())
        lines: List[str] = []
        for key, value in items.items():
            if isinstance(value, (list, tuple)) and len(value) == 3:
                formatted = _fmt_vector(value, digits)
            elif isinstance(value, (int, float, np.floating, np.integer)):
                formatted = _fmt_number(value, digits)
            else:
                formatted = str(value)
            lines.append(f"{str(key).ljust(width)} : {formatted}")
        record("\n".join(lines) + "\n")

    def record_list(title: str, rows: Iterable[str]) -> None:
        record(f"{title}\n")
        for row in rows:
            record(f"  - {row}\n")

    def record_json(title: str, value: Any) -> None:
        record(f"{title}\n")
        record(_pad_lines(_fmt_json(value)) + "\n")

    try:
        record_section("Bracket generation start")
        record_kv(
            {
                "Max bolt dia [mm]": max_bolt_dia,
                "Min edge clearance [mm]": min_edge_clearance,
                "Bolt spacing 1 [-]": bolt_spacing_1,
                "Bolt spacing 2 [-]": bolt_spacing_2,
                "Init bolt radius [mm]": init_bolt_radius,
                "Init plate thickness [mm]": init_plate_thickness,
                "Number of ribs": num_ribs,
                "Tolerance": tol,
                "Debug mode": debug,
            }
        )

        record_subsection("Material inputs")
        record_json("Free-side material", material_free)
        record_json("Bracket material", material_bracket)

        record_subsection("Free object inputs")
        record_json("Free objects", free_objects if free_objects else [])

        # ------------------------------------------------------------------
        # Planes / normals
        # ------------------------------------------------------------------
        record_section("Geometry setup")
        record_line("Computing surface planes and normalized surface normals.")

        plane_1 = geom.surface_plane(srf_1)
        plane_2 = geom.surface_plane(srf_2)
        perp_1 = plane_1.z_axis().unitized()
        perp_2 = plane_2.z_axis().unitized()

        plane_1 = geom.plane(plane_1.origin(), perp_1)
        plane_2 = geom.plane(plane_2.origin(), perp_2)

        record_kv(
            {
                "Plane 1 origin": _safe_serialize(plane_1.origin()),
                "Plane 2 origin": _safe_serialize(plane_2.origin()),
                "Plane 1 normal": _safe_serialize(perp_1),
                "Plane 2 normal": _safe_serialize(perp_2),
            }
        )

        # ------------------------------------------------------------------
        # Boundaries and boundary points
        # ------------------------------------------------------------------
        record_line("Extracting surface boundaries and boundary sample points.")
        boundary_1 = geom.surface_boundary(srf_1)
        boundary_2 = geom.surface_boundary(srf_2)
        boundary_pts_1 = geom.surface_boundary_points(srf_1)
        boundary_pts_2 = geom.surface_boundary_points(srf_2)

        record_kv(
            {
                "Boundary point count - surface 1": len(boundary_pts_1),
                "Boundary point count - surface 2": len(boundary_pts_2),
            }
        )

        # ------------------------------------------------------------------
        # Plane intersection line
        # ------------------------------------------------------------------
        record_section("Intersection and placement")
        record_line("Computing raw plane-plane intersection.")
        raw_intersection = geom.plane_plane_intersection(plane_1, plane_2)

        all_params_1 = [raw_intersection.closest_parameter(pt) for pt in boundary_pts_1]
        all_params_2 = [raw_intersection.closest_parameter(pt) for pt in boundary_pts_2]
        all_params = all_params_1 + all_params_2
        all_params_sorted = sorted(all_params)

        record_kv(
            {
                "Surface 1 param min": min(all_params_1),
                "Surface 1 param max": max(all_params_1),
                "Surface 2 param min": min(all_params_2),
                "Surface 2 param max": max(all_params_2),
                "Combined param min": all_params_sorted[0],
                "Combined param max": all_params_sorted[-1],
            }
        )

        intersect_line = geom.line(
            raw_intersection.point_at(all_params_sorted[0]),
            raw_intersection.point_at(all_params_sorted[-1]),
        )

        record_kv(
            {
                "Intersection line start": _safe_serialize(intersect_line.from_pt()),
                "Intersection line end": _safe_serialize(intersect_line.to_pt()),
            }
        )

        def get_far_edge(pts: List[GPoint], line: GLine) -> GLine:
            max_d = -1.0
            best: Optional[GLine] = None

            for pt in pts:
                t = line.closest_parameter(pt)
                cp = line.point_at(t)
                d = cp.distance_to(pt)
                if d > max_d:
                    max_d = d
                    best = geom.line(cp, pt)

            require(best is not None, "Could not compute far edge.")
            return best

        far_edge_1 = get_far_edge(boundary_pts_1, raw_intersection)
        far_edge_2 = get_far_edge(boundary_pts_2, raw_intersection)

        record_kv(
            {
                "Far edge 1 length [model units]": far_edge_1.length(),
                "Far edge 2 length [model units]": far_edge_2.length(),
                "Far edge 1 start": _safe_serialize(far_edge_1.from_pt()),
                "Far edge 1 end": _safe_serialize(far_edge_1.to_pt()),
                "Far edge 2 start": _safe_serialize(far_edge_2.from_pt()),
                "Far edge 2 end": _safe_serialize(far_edge_2.to_pt()),
            }
        )

        all_params_1_sorted = sorted(all_params_1)
        all_params_2_sorted = sorted(all_params_2)

        intervals_1 = [[all_params_1_sorted[0], all_params_1_sorted[-1]]]
        intervals_2 = [[all_params_2_sorted[0], all_params_2_sorted[-1]]]

        def find_overlapping_regions(a: List[Interval], b: List[Interval]) -> List[Interval]:
            a_sorted = sorted(a, key=lambda x: x[0])
            b_sorted = sorted(b, key=lambda x: x[0])

            i = j = 0
            overlaps: List[Interval] = []

            while i < len(a_sorted) and j < len(b_sorted):
                a_start, a_end = a_sorted[i]
                b_start, b_end = b_sorted[j]
                start = max(a_start, b_start)
                end = min(a_end, b_end)

                if start < end:
                    overlaps.append((start, end))

                if a_end < b_end:
                    i += 1
                else:
                    j += 1

            return overlaps

        overlaps = find_overlapping_regions(intervals_1, intervals_2)

        record_json(
            "Overlap search details",
            {
                "intervals_surface_1": intervals_1,
                "intervals_surface_2": intervals_2,
                "overlaps": overlaps,
            },
        )

        require(
            len(overlaps) > 0,
            "No overlapping interval found between the two boundaries along the intersection line.",
        )

        placement_edge = geom.line(
            raw_intersection.point_at(overlaps[0][0]),
            raw_intersection.point_at(overlaps[0][1]),
        )
        mid = raw_intersection.point_at((overlaps[0][0] + overlaps[0][1]) / 2.0)

        record_kv(
            {
                "Placement edge start": _safe_serialize(placement_edge.from_pt()),
                "Placement edge end": _safe_serialize(placement_edge.to_pt()),
                "Placement midpoint": _safe_serialize(mid),
            }
        )

        # ------------------------------------------------------------------
        # Mid extension logic
        # ------------------------------------------------------------------
        def mid_extension(
            ref_edge: GLine,
            placement: GLine,
            boundary: GCurve,
            mid_pt: GPoint,
        ) -> GLine:
            v1 = geom.vector(
                placement.from_pt().x - ref_edge.from_pt().x,
                placement.from_pt().y - ref_edge.from_pt().y,
                placement.from_pt().z - ref_edge.from_pt().z,
            )
            v2 = geom.vector(
                placement.to_pt().x - ref_edge.from_pt().x,
                placement.to_pt().y - ref_edge.from_pt().y,
                placement.to_pt().z - ref_edge.from_pt().z,
            )

            p1 = ref_edge.translated(v1)
            p2 = ref_edge.translated(v2)

            tB1 = geom.curve_line_intersection(boundary, p1)
            tB2 = geom.curve_line_intersection(boundary, p2)

            require(len(tB1) > 0, "No CurveLine intersections for placement_1.")
            require(len(tB2) > 0, "No CurveLine intersections for placement_2.")

            extension_p = max(max(tB1), max(tB2))
            extension_pt = ref_edge.point_at(extension_p)

            ext = geom.line(ref_edge.from_pt(), extension_pt)
            return ext.translated(
                geom.vector(
                    mid_pt.x - ref_edge.from_pt().x,
                    mid_pt.y - ref_edge.from_pt().y,
                    mid_pt.z - ref_edge.from_pt().z,
                )
            )

        record_line("Computing mid-extension lines that project away from the intersection.")
        mid_extension_1 = mid_extension(far_edge_1, placement_edge, boundary_1, mid)
        mid_extension_2 = mid_extension(far_edge_2, placement_edge, boundary_2, mid)

        record_kv(
            {
                "Mid extension 1 length [model units]": mid_extension_1.length(),
                "Mid extension 2 length [model units]": mid_extension_2.length(),
                "Mid extension 1 start": _safe_serialize(mid_extension_1.from_pt()),
                "Mid extension 1 end": _safe_serialize(mid_extension_1.to_pt()),
                "Mid extension 2 start": _safe_serialize(mid_extension_2.from_pt()),
                "Mid extension 2 end": _safe_serialize(mid_extension_2.to_pt()),
            }
        )

        # ------------------------------------------------------------------
        # Bracket dims / axes
        # ------------------------------------------------------------------
        record_section("Bracket layout and bolt placement")

        half_bracket_width = max_bolt_dia * 5.0
        bracket_width = half_bracket_width * 2.0
        quarter_bracket_width = half_bracket_width / 2.0

        axis_0 = placement_edge.direction().unitized()
        axis_1 = mid_extension_1.direction().unitized()
        axis_2 = mid_extension_2.direction().unitized()

        record_kv(
            {
                "Axis 0": _safe_serialize(axis_0),
                "Axis 1": _safe_serialize(axis_1),
                "Axis 2": _safe_serialize(axis_2),
                "Half bracket width [model units]": half_bracket_width,
                "Quarter bracket width [model units]": quarter_bracket_width,
                "Bracket width [model units]": bracket_width,
            }
        )

        def curve_intervals_inside_poly_on_line(
            line: GLine,
            poly_crv: GCurve,
            plane: GPlane,
            tol: float,
        ) -> List[Interval]:
            pts = poly_crv.to_points()
            require(len(pts) >= 2, "Polygon curve has too few points.")

            params = [0.0, 1.0]

            for i, a in enumerate(pts[:-1]):
                edge = geom.line(a, pts[i + 1])
                success, p_line, p_edge = geom.line_line_params(line, edge)
                if not success:
                    continue
                if (-tol <= p_edge <= 1.0 + tol) and (-tol <= p_line <= 1.0 + tol):
                    params.append(p_line)

            final_params = unique_floats(params, tol)
            require(len(final_params) >= 2, "Failed to compute usable intersection params.")

            intervals: List[Interval] = []
            for k in range(len(final_params) - 1):
                mid_param = (final_params[k] + final_params[k + 1]) / 2.0
                mid_pt = line.point_at(mid_param)
                if poly_crv.contains(mid_pt, plane) == "inside":
                    intervals.append((final_params[k], final_params[k + 1]))

            return intervals

        def near_far_distances_on_line_for_inside_interval(
            line: GLine,
            inside: Interval,
        ) -> Tuple[float, float]:
            a0 = line.point_at(0.0)
            near = a0.distance_to(line.point_at(inside[0]))
            far = a0.distance_to(line.point_at(inside[1]))
            return near, far

        def offset_lines(base_line: GLine, axis: GVector, offset: float) -> List[GLine]:
            return [
                base_line.translated(axis.scaled(-offset)),
                base_line.translated(axis.scaled(+offset)),
            ]

        def near_far_for_two_offsets(
            base_line: GLine,
            boundary: GCurve,
            plane: GPlane,
        ) -> Tuple[List[float], List[float]]:
            nears: List[float] = []
            fars: List[float] = []

            for ln in offset_lines(base_line, axis_0, quarter_bracket_width):
                inside = curve_intervals_inside_poly_on_line(ln, boundary, plane, tol)
                require(len(inside) > 0, "Failed to compute inside interval for near/far calc.")
                n, f = near_far_distances_on_line_for_inside_interval(ln, inside[0])
                nears.append(n)
                fars.append(f)

            return nears, fars

        record_subsection("Offset-line edge distances")
        near_1, far_1 = near_far_for_two_offsets(mid_extension_1, boundary_1, plane_1)
        near_2, far_2 = near_far_for_two_offsets(mid_extension_2, boundary_2, plane_2)

        record_kv(
            {
                "Surface 1 near values": near_1,
                "Surface 1 far values": far_1,
                "Surface 2 near values": near_2,
                "Surface 2 far values": far_2,
            }
        )

        # ------------------------------------------------------------------
        # Bolt centerline distances
        # ------------------------------------------------------------------
        first_far_1 = sorted(far_1)[0]
        last_near_1 = sorted(near_1)[-1]
        cp_bolts_1_1 = last_near_1 + min_edge_clearance
        cp_bolts_1_2 = min(first_far_1, cp_bolts_1_1 + half_bracket_width * bolt_spacing_1)

        first_far_2 = sorted(far_2)[0]
        last_near_2 = sorted(near_2)[-1]
        cp_bolts_2_1 = last_near_2 + min_edge_clearance
        cp_bolts_2_2 = min(first_far_2, cp_bolts_2_1 + half_bracket_width * bolt_spacing_2)

        record_subsection("Bolt centerline distances")
        record_kv(
            {
                "Surface 1 first far [model units]": first_far_1,
                "Surface 1 last near [model units]": last_near_1,
                "Surface 1 cp_bolts_1_1 [model units]": cp_bolts_1_1,
                "Surface 1 cp_bolts_1_2 [model units]": cp_bolts_1_2,
                "Surface 2 first far [model units]": first_far_2,
                "Surface 2 last near [model units]": last_near_2,
                "Surface 2 cp_bolts_2_1 [model units]": cp_bolts_2_1,
                "Surface 2 cp_bolts_2_2 [model units]": cp_bolts_2_2,
            }
        )

        # ------------------------------------------------------------------
        # Bolt circles / points
        # ------------------------------------------------------------------
        def bolt_circles_and_points(
            plane: GPlane,
            axis_long: GVector,
            cp1: float,
            cp2: float,
        ):
            cps: List[GPoint] = []
            for i in (quarter_bracket_width, -quarter_bracket_width):
                for j in (cp1, cp2):
                    p = mid.translated(axis_0.scaled(i)).translated(axis_long.scaled(j))
                    cps.append(p)

            circles: List[GCircle] = [geom.circle(plane, cp, init_bolt_radius) for cp in cps]
            return circles, cps

        bolt_circles_1, cps_bolts_1 = bolt_circles_and_points(
            plane_1, axis_1, cp_bolts_1_1, cp_bolts_1_2
        )
        bolt_circles_2, cps_bolts_2 = bolt_circles_and_points(
            plane_2, axis_2, cp_bolts_2_1, cp_bolts_2_2
        )

        record_subsection("Bolt points")
        record_kv(
            {
                "Bolt count - surface 1": len(cps_bolts_1),
                "Bolt count - surface 2": len(cps_bolts_2),
                "Init bolt radius [model units]": init_bolt_radius,
            }
        )
        record_json("Surface 1 bolt points", cps_bolts_1)
        record_json("Surface 2 bolt points", cps_bolts_2)

        # ------------------------------------------------------------------
        # Bolt solids
        # ------------------------------------------------------------------
        bolt_extension = 10.0

        def bolt_solids(circles: List[GCircle], perp: GVector) -> List[GSolid]:
            out: List[GSolid] = []
            for c in circles:
                crv = c.get_curve()
                shifted = crv.translated(perp.unitized().scaled(-bolt_extension))
                out.append(
                    geom.extrusion(
                        shifted,
                        perp.unitized(),
                        bolt_extension * 2.0 + init_plate_thickness,
                    )
                )
            return out

        bolts_geo: List[GSolid] = []
        bolts_geo += bolt_solids(bolt_circles_1, perp_1)
        bolts_geo += bolt_solids(bolt_circles_2, perp_2)

        record_subsection("Bolt solid geometry")
        record_kv(
            {
                "Bolt extension [model units]": bolt_extension,
                "Bolt solid count": len(bolts_geo),
            }
        )

        # ------------------------------------------------------------------
        # Bracket profile
        # ------------------------------------------------------------------
        def make_bracket_profile(
            geom: GeomBackend,
            mid: GPoint,
            axis_0: GVector,
            axis_1: GVector,
            axis_2: GVector,
            perp_1: GVector,
            perp_2: GVector,
            half_bracket_width: float,
            quarter_bracket_width: float,
            cp_bolts_1_2: float,
            cp_bolts_2_2: float,
            init_plate_thickness: float,
        ) -> Dict[str, Any]:
            corner_pt = mid.translated(axis_0.unitized().scaled(-half_bracket_width))

            corner_pts: List[GPoint] = []
            for i, d in enumerate([axis_1, axis_2]):
                dist = ([cp_bolts_1_2, cp_bolts_2_2][i] + quarter_bracket_width)
                v = d.unitized().scaled(dist)
                p = corner_pt.translated(v)
                corner_pts.append(p)

            corner_pts.insert(1, corner_pt)

            norm_1 = perp_1.unitized().scaled(init_plate_thickness)
            norm_2 = perp_2.unitized().scaled(init_plate_thickness)
            norm_mid = norm_1 + norm_2

            for i in range(3):
                if i == 0:
                    new_pt = corner_pts[2].translated(norm_2)
                elif i == 1:
                    new_pt = corner_pts[1].translated(norm_mid)
                else:
                    new_pt = corner_pts[0].translated(norm_1)
                corner_pts.append(new_pt)

            profile_curve = geom.curve(corner_pts + [corner_pts[0]])
            inside_poly_curve = geom.curve(corner_pts[-3:] + [corner_pts[-3]])

            return {
                "profile_curve": profile_curve,
                "inside_poly_curve": inside_poly_curve,
                "corner_pts": corner_pts,
            }

        record_section("Bracket solid generation")

        prof = make_bracket_profile(
            geom=geom,
            mid=mid,
            axis_0=axis_0,
            axis_1=axis_1,
            axis_2=axis_2,
            perp_1=perp_1,
            perp_2=perp_2,
            half_bracket_width=half_bracket_width,
            quarter_bracket_width=quarter_bracket_width,
            cp_bolts_1_2=cp_bolts_1_2,
            cp_bolts_2_2=cp_bolts_2_2,
            init_plate_thickness=init_plate_thickness,
        )

        profile_curve = prof["profile_curve"]
        inside_poly_curve = prof["inside_poly_curve"]

        record_json("Bracket profile corner points", prof["corner_pts"])

        bracket_geo = geom.extrusion(profile_curve, axis_0.unitized(), bracket_width)
        record_line("Created primary bracket extrusion.")

        bracket_geo_bool = bracket_geo
        for i, other_geo in enumerate(bolts_geo, start=1):
            record_line(f"Applying bolt-hole boolean difference {i}/{len(bolts_geo)}.")
            bracket_geo_bool = geom.boolean_difference(bracket_geo_bool, other_geo)

        # ------------------------------------------------------------------
        # Ribs
        # ------------------------------------------------------------------
        if num_ribs == 1:
            facs = [0.5]
        elif num_ribs == 2:
            facs = [0.0, 1.0]
        else:
            facs = []

        inside_ribs: List[GSolid] = []
        for fac in facs:
            rib = geom.extrusion(
                inside_poly_curve,
                axis_0.unitized(),
                init_plate_thickness,
            )
            shift = bracket_width * fac - init_plate_thickness * fac
            inside_ribs.append(rib.translated(axis_0.scaled(shift)))

        record_kv(
            {
                "Requested ribs": num_ribs,
                "Generated rib count": len(inside_ribs),
                "Rib placement factors": facs,
            }
        )

        bracket_solid = geom.boolean_union([bracket_geo_bool] + inside_ribs)
        record_line("Bracket solid boolean union completed.")

        # ------------------------------------------------------------------
        # Optional engineering calculations
        # ------------------------------------------------------------------
        mass_info: Dict[str, Any] = {}
        reaction_info: Dict[str, Any] = {}
        bolt_info: Dict[str, Any] = {}
        opt_info: Dict[str, Any] = {}

        if material_free and material_bracket and free_objects and len(free_objects) > 0:
            record_section("Engineering calculations")

            free_density = _material_value(material_free, "Density [kg/m?]", "Density")
            bracket_density = _material_value(material_bracket, "Density [kg/m?]", "Density")

            free_volume = (
                sum(obj.get("volume", 0.0) or 0.0 for obj in free_objects) / 1_000_000_000.0
            )
            free_mass = free_volume * free_density

            bracket_volume = bracket_solid.volume() / 1_000_000_000.0
            bracket_mass = bracket_volume * bracket_density

            combined_mass = free_mass + bracket_mass
            gravity_force_1 = free_mass * 9.81
            gravity_force_2 = combined_mass * 9.81

            free_cg_raw = free_objects[0].get("cg")
            require(
                free_cg_raw is not None and len(free_cg_raw) == 3,
                "free_objects[0]['cg'] must be a 3-item coordinate list.",
            )

            free_cg = geom.point(*free_cg_raw)
            bracket_cg = bracket_solid.centroid()

            combined_cg = geom.point(
                (free_mass * free_cg.x + bracket_mass * bracket_cg.x) / combined_mass,
                (free_mass * free_cg.y + bracket_mass * bracket_cg.y) / combined_mass,
                (free_mass * free_cg.z + bracket_mass * bracket_cg.z) / combined_mass,
            )

            mass_info = {
                "free_density": free_density,
                "bracket_density": bracket_density,
                "free_volume_m3": free_volume,
                "bracket_volume_m3": bracket_volume,
                "free_mass": free_mass,
                "bracket_mass": bracket_mass,
                "combined_mass": combined_mass,
                "gravity_force_1": gravity_force_1,
                "gravity_force_2": gravity_force_2,
                "free_cg": free_cg,
                "bracket_cg": bracket_cg,
                "combined_cg": combined_cg,
            }

            record_subsection("Mass properties")
            record_kv(
                {
                    "Free density [kg/m^3]": free_density,
                    "Bracket density [kg/m^3]": bracket_density,
                    "Free volume [m^3]": free_volume,
                    "Bracket volume [m^3]": bracket_volume,
                    "Free mass [kg]": free_mass,
                    "Bracket mass [kg]": bracket_mass,
                    "Combined mass [kg]": combined_mass,
                    "Gravity force 1 [N]": gravity_force_1,
                    "Gravity force 2 [N]": gravity_force_2,
                    "Free CG [model units]": _safe_serialize(free_cg),
                    "Bracket CG [model units]": _safe_serialize(bracket_cg),
                    "Combined CG [model units]": _safe_serialize(combined_cg),
                }
            )

            def compute_reactions(
                calculate_equilibrium_reactions,
                vector_magnitude,
                applied_F,
                applied_M,
                reaction_pts_xyz_m,
                cg_xyz_m,
                case_name: str,
            ):
                record_subsection(f"Reaction calculation - {case_name}")
                record_kv(
                    {
                        "Applied Fx [N]": applied_F[0],
                        "Applied Fy [N]": applied_F[1],
                        "Applied Fz [N]": applied_F[2],
                        "Applied Mx [Nm]": applied_M[0],
                        "Applied My [Nm]": applied_M[1],
                        "Applied Mz [Nm]": applied_M[2],
                        "CG [m]": cg_xyz_m,
                    }
                )
                record_json("Reaction points [m]", reaction_pts_xyz_m)

                reactions = calculate_equilibrium_reactions(
                    applied_F,
                    applied_M,
                    reaction_pts_xyz_m,
                    cg_xyz_m,
                )

                rows = []
                for i, (pt, r) in enumerate(zip(reaction_pts_xyz_m, reactions), start=1):
                    mag = float(vector_magnitude(r))
                    rows.append(
                        f"Support {i}: point={_fmt_vector(pt)} | "
                        f"R={_fmt_vector(_safe_serialize(r))} | |R|={mag:.3f} N"
                    )

                record_list("Support reactions", rows)
                return reactions

            def reactions_to_T_S_vectors(
                split_vector,
                reactions_xyz,
                perp_vec_xyz,
                case_name: str,
            ):
                record_subsection(f"Reaction decomposition - {case_name}")
                record_kv({"Perpendicular reference vector": perp_vec_xyz})

                T, S = [], []
                rows = []

                for i, reaction in enumerate(reactions_xyz, start=1):
                    t, s = split_vector(reaction, perp_vec_xyz)
                    T.append(t)
                    S.append(s)

                    t_mag = float(np.linalg.norm(t))
                    s_mag = float(np.linalg.norm(s))

                    rows.append(
                        f"Bolt {i}: "
                        f"T={_fmt_vector(_safe_serialize(t))} | |T|={t_mag:.3f} N | "
                        f"S={_fmt_vector(_safe_serialize(s))} | |S|={s_mag:.3f} N"
                    )

                record_list("Axial / shear split", rows)
                return T, S

            record_subsection("Importing external solvers")
            from solvers.rigid_body_solver import (
                calculate_equilibrium_reactions,
                split_vector,
                vector_magnitude,
            )
            from solvers.bolt_sizing import select_bolt_gh
            from solvers.section_sizing import optimum_thickness

            # --------------------------------------------------------------
            # Reactions: case 2 (free object load)
            # --------------------------------------------------------------
            applied_F = [0, 0, gravity_force_1]
            applied_M = [0, 0, 0]

            _free_cg = free_cg._serialize()
            cg_free = [_free_cg[0] / 1000.0, _free_cg[1] / 1000.0, _free_cg[2] / 1000.0]

            _cps_bolts_2 = [pt._serialize() for pt in cps_bolts_2]
            reaction_points_2 = [
                [pt[0] / 1000.0, pt[1] / 1000.0, pt[2] / 1000.0]
                for pt in _cps_bolts_2
            ]

            R_2 = compute_reactions(
                calculate_equilibrium_reactions,
                vector_magnitude,
                applied_F,
                applied_M,
                reaction_points_2,
                cg_free,
                case_name="Case 2 (free object load on bracket-side bolts)",
            )

            _perp_2 = perp_2._serialize()
            T_2, S_2 = reactions_to_T_S_vectors(
                split_vector,
                R_2,
                [_perp_2[0], _perp_2[1], _perp_2[2]],
                case_name="Case 2",
            )

            # --------------------------------------------------------------
            # Bolt sizing: case 2
            # --------------------------------------------------------------
            record_subsection("Bolt sizing - Case 2")
            bolt_summary: List[Any] = []
            bolt_sizes: List[Any] = []
            bolt_hole_sizes: List[Any] = []

            case_2_rows = []
            for i, (T, S) in enumerate(zip(T_2, S_2), start=1):
                axial_mag = float(np.linalg.norm(T))
                shear_mag = float(np.linalg.norm(S))

                result, summary = select_bolt_gh(axial_mag, shear_mag)

                bolt_summary.append(summary)
                bolt_sizes.append(result.size)
                bolt_hole_sizes.append(result.hole_size)

                case_2_rows.append(
                    f"Bolt {i}: |T|={axial_mag:.3f} N, "
                    f"|S|={shear_mag:.3f} N, "
                    f"selected size={result.size}, hole size={result.hole_size}, "
                    f"summary={summary}"
                )

            hole_size_2 = max(bolt_hole_sizes)
            record_list("Per-bolt sizing results", case_2_rows)
            record_kv({"Governing hole size - Case 2": hole_size_2})

            # --------------------------------------------------------------
            # Reactions: case 1 (combined mass on fixed side)
            # --------------------------------------------------------------
            applied_F = [0, 0, gravity_force_2]
            applied_M = [0, 0, 0]

            _combined_cg = combined_cg._serialize()
            cg_combined = [
                _combined_cg[0] / 1000.0,
                _combined_cg[1] / 1000.0,
                _combined_cg[2] / 1000.0,
            ]

            _cps_bolts_1 = [pt._serialize() for pt in cps_bolts_1]
            reaction_points_1 = [
                [pt[0] / 1000.0, pt[1] / 1000.0, pt[2] / 1000.0]
                for pt in _cps_bolts_1
            ]

            R_1 = compute_reactions(
                calculate_equilibrium_reactions,
                vector_magnitude,
                applied_F,
                applied_M,
                reaction_points_1,
                cg_combined,
                case_name="Case 1 (combined mass on fixed-side bolts)",
            )

            _perp_1 = perp_1._serialize()
            T_1, S_1 = reactions_to_T_S_vectors(
                split_vector,
                R_1,
                [_perp_1[0], _perp_1[1], _perp_1[2]],
                case_name="Case 1",
            )

            reaction_info = {
                "R_1": R_1.tolist(),
                "T_1": [l.tolist() for l in T_1],
                "S_1": [l.tolist() for l in S_1],
                "R_2": R_2.tolist(),
                "T_2": [l.tolist() for l in T_2],
                "S_2": [l.tolist() for l in S_2],
            }

            record_json("Serialized reaction output payload", reaction_info)

            # --------------------------------------------------------------
            # Bolt sizing: case 1
            # --------------------------------------------------------------
            record_subsection("Bolt sizing - Case 1")
            case_1_rows = []
            new_bolt_hole_sizes = []

            for i, (T, S) in enumerate(zip(T_1, S_1), start=1):
                axial_mag = float(np.linalg.norm(T))
                shear_mag = float(np.linalg.norm(S))

                result, summary = select_bolt_gh(axial_mag, shear_mag)

                bolt_summary.append(summary)
                bolt_sizes.append(result.size)
                new_bolt_hole_sizes.append(result.hole_size)

                case_1_rows.append(
                    f"Bolt {i}: |T|={axial_mag:.3f} N, "
                    f"|S|={shear_mag:.3f} N, "
                    f"selected size={result.size}, hole size={result.hole_size}, "
                    f"summary={summary}"
                )

            hole_size_1 = max(new_bolt_hole_sizes)
            bolt_hole_sizes += new_bolt_hole_sizes

            bolt_info = {
                "bolt_summary": bolt_summary,
                "bolt_sizes": bolt_sizes,
                "bolt_hole_sizes": bolt_hole_sizes,
                "hole_size_1": hole_size_1,
                "hole_size_2": hole_size_2,
            }

            record_list("Per-bolt sizing results", case_1_rows)
            record_kv(
                {
                    "Governing hole size - Case 1": hole_size_1,
                    "Governing hole size - Case 2": hole_size_2,
                }
            )
            record_json("Serialized bolt output payload", bolt_info)

            # --------------------------------------------------------------
            # Plate thickness optimization
            # --------------------------------------------------------------
            record_section("Plate thickness optimization")

            record_line(
                "Assumption: Case 2 axial reactions are used to derive the bending moment "
                "for bracket section sizing."
            )

            F1 = geom.vector(*T_2[1]).length() + geom.vector(*T_2[3]).length()
            F2 = geom.vector(*T_2[0]).length() + geom.vector(*T_2[2]).length()
            L1 = cp_bolts_2_2 / 1000.0
            L2 = cp_bolts_2_1 / 1000.0
            M = (F1 * L1) - (F2 * L2)

            bb = bracket_solid.bounding_box(plane_2)
            bb_height = bb["max"][2] - bb["min"][2]

            bracket_yield_strength = _material_value(
                material_bracket,
                "Yield Strength [MPa]",
                "Yield Strength",
            )

            b = bracket_width / 1000.0
            h = bb_height / 1000.0
            s = bracket_yield_strength * 1_000_000.0

            record_kv(
                {
                    "F1 [N]": F1,
                    "F2 [N]": F2,
                    "L1 [m]": L1,
                    "L2 [m]": L2,
                    "Bending moment M [Nm]": M,
                    "Bracket width b [m]": b,
                    "Section height h [m]": h,
                    "Bracket yield strength [MPa]": bracket_yield_strength,
                    "Bracket yield strength [Pa]": s,
                    "Bounding-box height [model units]": bb_height,
                }
            )
            record_json("Bounding box", bb)

            t_opt = optimum_thickness(b, h, M, s) * 1000.0

            opt_info = {
                "bb": bb,
                "t_opt": t_opt,
            }

            record_kv({"Recommended thickness [mm]": t_opt})
            record_json("Serialized optimization output payload", opt_info)
        else:
            record_section("Engineering calculations skipped")
            record_line(
                "Skipped because one or more required inputs were missing: "
                "material_free, material_bracket, free_objects."
            )

        record_section("Bracket generation complete")

        return {
            "plane_1": plane_1,
            "plane_2": plane_2,
            "perp_1": perp_1,
            "perp_2": perp_2,
            "boundary_1": boundary_1,
            "boundary_2": boundary_2,
            "raw_intersection": raw_intersection,
            "intersect_line": intersect_line,
            "far_edge_1": far_edge_1,
            "far_edge_2": far_edge_2,
            "placement_edge": placement_edge,
            "mid_extension_1": mid_extension_1,
            "mid_extension_2": mid_extension_2,
            "axis_0": axis_0,
            "axis_1": axis_1,
            "axis_2": axis_2,
            "half_bracket_width": half_bracket_width,
            "quarter_bracket_width": quarter_bracket_width,
            "bracket_width": bracket_width,
            "cp_bolts_1_1": cp_bolts_1_1,
            "cp_bolts_1_2": cp_bolts_1_2,
            "cp_bolts_2_1": cp_bolts_2_1,
            "cp_bolts_2_2": cp_bolts_2_2,
            "cps_bolts_1": cps_bolts_1,
            "cps_bolts_2": cps_bolts_2,
            "profile_curve": profile_curve,
            "inside_poly_curve": inside_poly_curve,
            "bracket_solid": bracket_solid,
            "inside_ribs": inside_ribs,
            "mass": mass_info,
            "reactions": reaction_info,
            "bolts": bolt_info,
            "opt": opt_info,
            "log": log_record,
        }

    except Exception as e:
        record_section("Bracket generation failed")
        record_kv(
            {
                "Error type": type(e).__name__,
                "Error message": str(e),
            }
        )
        raise RuntimeError(f"{str(e)}\n\n--- bracket log ---\n{log_record}") from e