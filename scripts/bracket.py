# bracket/bracket_script.py
from __future__ import annotations
import math

from typing import Any, Callable, Dict, List, Optional, Tuple

from helpers import log
from geom.common import TOL, require, unique_floats
from geom.protocol import GeomBackend, GCircle, GCurve, GLine, GPlane, GPoint, GVector, GSolid, Interval

import numpy as np

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

    material_dictionary: Optional[Dict[str, Dict[str, str]]] = None,
    material_free: Optional[str] = None,
    material_bracket: Optional[str] = None,

    free_objects: Optional[List[GSolid]] = None,

    tol: float = TOL,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Bracket generator.

    - srf_1, srf_2: backend surface/face objects understood by `geom.surface_*`
    - returns OCC/Rhino/etc backend geometry objects depending on the backend
    """

    # ---- planes / normals
    plane_1 = geom.surface_plane(srf_1)
    plane_2 = geom.surface_plane(srf_2)
    perp_1 = plane_1.z_axis().unitized()
    perp_2 = plane_2.z_axis().unitized()

    (perp_1, perp_2) = geom.normalize_vector_pair(perp_1, perp_2)
    
    plane_1 = geom.plane(plane_1.origin(), perp_1)
    plane_2 = geom.plane(plane_2.origin(), perp_2)

    # ---- boundaries and boundary points
    boundary_1 = geom.surface_boundary(srf_1)
    boundary_2 = geom.surface_boundary(srf_2)
    boundary_pts_1 = geom.surface_boundary_points(srf_1)
    boundary_pts_2 = geom.surface_boundary_points(srf_2)

    # ---- plane intersection line
    raw_intersection = geom.plane_plane_intersection(plane_1, plane_2)

    # ---- clip intersection line to extrema of both boundaries (in param space)
    all_params_1 = [raw_intersection.closest_parameter(pt) for pt in boundary_pts_1]
    all_params_2 = [raw_intersection.closest_parameter(pt) for pt in boundary_pts_2]

    all_params = all_params_1 + all_params_2
    all_params_sorted = sorted(all_params)

    intersect_line = geom.line(
        raw_intersection.point_at(all_params_sorted[0]),
        raw_intersection.point_at(all_params_sorted[-1]),
    )

    def get_far_edge(
        pts: List[GPoint], 
        line: GLine
    ) -> GLine:
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

    # ---- offset boundaries away from the intersection using far_edge directions
    # poly_1 = boundary_1.duplicate().translated(
    #     far_edge_1.direction().unitized().scaled(-min_bracket_extension)
    # )
    # poly_2 = boundary_2.duplicate().translated(
    #     far_edge_2.direction().unitized().scaled(-min_bracket_extension)
    # )

    # ---- overlap interval along intersection line
    # intervals_1 = curve_intervals_inside_poly_on_line(
    #     geom, intersect_line, poly_1, plane_1, tol
    # )
    # intervals_2 = curve_intervals_inside_poly_on_line(
    #     geom, intersect_line, poly_2, plane_2, tol
    # )

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
    require(
        len(overlaps) > 0,
        "No overlapping interval found between the two boundaries along the intersection line.",
    )

    placement_edge = geom.line(raw_intersection.point_at(overlaps[0][0]), raw_intersection.point_at(overlaps[0][1]))
    mid = raw_intersection.point_at((overlaps[0][0] + overlaps[0][1]) / 2.0)

    # ---- mid extension logic (uses CurveLine intersections)
    def mid_extension(ref_edge: GLine, placement: GLine, boundary: GCurve, mid_pt: GPoint) -> GLine:
        # translate ref_edge so its From coincides with each placement endpoint
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

    mid_extension_1 = mid_extension(far_edge_1, placement_edge, boundary_1, mid)
    mid_extension_2 = mid_extension(far_edge_2, placement_edge, boundary_2, mid)

    # ---- bracket dims / axes
    half_bracket_width = max_bolt_dia * 5.0
    bracket_width = half_bracket_width * 2.0
    quarter_bracket_width = half_bracket_width / 2.0

    axis_0 = placement_edge.direction().unitized()
    axis_1 = mid_extension_1.direction().unitized()
    axis_2 = mid_extension_2.direction().unitized()
    
    # ---- compute near/far on two offset lines for each side

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
            # only add intersections within physical lines [0, 1] domain
            if (-tol <= p_edge <= 1.0 + tol) and (-tol <= p_line <= 1.0 + tol):
                params.append(p_line)

        final_params = unique_floats(params, tol)
        require(len(final_params) >= 2, "Failed to compute usable intersection params.")

        # return parameter intervals that are inside the poly_crv shape
        intervals: List[Interval] = []
        for k in range(len(final_params) - 1):
            mid = (final_params[k] + final_params[k + 1]) / 2.0
            mid_pt = line.point_at(mid)
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

    def near_far_for_two_offsets(base_line: GLine, boundary: GCurve, plane: GPlane) -> Tuple[List[float], List[float]]:
        nears: List[float] = []
        fars: List[float] = []
        for ln in offset_lines(base_line, axis_0, quarter_bracket_width):
            inside = curve_intervals_inside_poly_on_line(ln, boundary, plane, tol)
            require(len(inside) > 0, "Failed to compute inside interval for near/far calc.")
            n, f = near_far_distances_on_line_for_inside_interval(ln, inside[0])
            nears.append(n)
            fars.append(f)
        return nears, fars

    log("Starting mid extension 1")
    near_1, far_1 = near_far_for_two_offsets(mid_extension_1, boundary_1, plane_1)
    log("Starting mid extension 2")
    near_2, far_2 = near_far_for_two_offsets(mid_extension_2, boundary_2, plane_2)

    # ---- bolt centerline distances
    first_far_1 = sorted(far_1)[0]
    last_near_1 = sorted(near_1)[-1]
    cp_bolts_1_1 = last_near_1 + min_edge_clearance
    cp_bolts_1_2 = min(first_far_1, cp_bolts_1_1 + half_bracket_width * bolt_spacing_1)

    first_far_2 = sorted(far_2)[0]
    last_near_2 = sorted(near_2)[-1]
    cp_bolts_2_1 = last_near_2 + min_edge_clearance
    cp_bolts_2_2 = min(first_far_2, cp_bolts_2_1 + half_bracket_width * bolt_spacing_2)

    # ---- bolt circles / points
    def bolt_circles_and_points(plane: GPlane, axis_long: GVector, cp1: float, cp2: float):
        cps: List[GPoint] = []
        for i in (quarter_bracket_width, -quarter_bracket_width):
            for j in (cp1, cp2):
                p = mid.translated(axis_0.scaled(i)).translated(axis_long.scaled(j))
                cps.append(p)
        circles: List[GCircle] = [geom.circle(plane, cp, init_bolt_radius) for cp in cps]
        return circles, cps

    bolt_circles_1, cps_bolts_1 = bolt_circles_and_points(plane_1, axis_1, cp_bolts_1_1, cp_bolts_1_2)
    bolt_circles_2, cps_bolts_2 = bolt_circles_and_points(plane_2, axis_2, cp_bolts_2_1, cp_bolts_2_2)

    # ---- bolt solids (simple extrusions)
    bolt_extension = 10.0

    def bolt_solids(circles: List[GCircle], perp: GVector) -> List[GSolid]:
        out: List[GSolid] = []
        for c in circles:
            crv = c.get_curve()
            shifted = crv.translated(perp.unitized().scaled(-bolt_extension))
            out.append(geom.extrusion(shifted, perp.unitized(), bolt_extension * 2.0 + init_plate_thickness))
        return out

    bolts_geo: List[GSolid] = []
    bolts_geo += bolt_solids(bolt_circles_1, perp_1)
    bolts_geo += bolt_solids(bolt_circles_2, perp_2)

    # ---- bracket profile

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

        # norm_1 = axis_1.rotated(math.pi/2, axis_0).unitized().scaled(init_plate_thickness)
        # norm_2 = axis_2.rotated(-math.pi/2, axis_0).unitized().scaled(init_plate_thickness)
        norm_1 = perp_1.unitized().scaled(init_plate_thickness)
        norm_2 = perp_2.unitized().scaled(init_plate_thickness)

        norm_mid = norm_1 + norm_2

        for i in range(3):
            if i == 0:
                new_pt = corner_pts[2].translated(norm_2)
            if i == 1:
                new_pt = corner_pts[1].translated(norm_mid)
            if i == 2:
                new_pt = corner_pts[0].translated(norm_1)
            corner_pts.append(new_pt)

        profile_curve = geom.curve(corner_pts + [corner_pts[0]])
        inside_poly_curve = geom.curve(corner_pts[-3:] + [corner_pts[-3]])

        return {
            "profile_curve": profile_curve,
            "inside_poly_curve": inside_poly_curve,
        }

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

    bracket_geo = geom.extrusion(profile_curve, axis_0.unitized(), bracket_width)
    
    bracket_geo_bool = bracket_geo
    for other_geo in bolts_geo:
        bracket_geo_bool = geom.boolean_difference(bracket_geo_bool, other_geo)

    # ---- ribs
    if num_ribs == 1:
        facs = [0.5]
    elif num_ribs == 2:
        facs = [0.0, 1.0]
    else:
        facs = []
    
    inside_ribs: List[GSolid] = []
    for fac in facs:
        rib = geom.extrusion(inside_poly_curve, axis_0.unitized(), init_plate_thickness)
        shift = bracket_width * fac - init_plate_thickness * fac
        inside_ribs.append(rib.translated(axis_0.scaled(shift)))

    bracket_solid = geom.boolean_union([bracket_geo_bool] + inside_ribs)

    # ---- optional mass properties (only if caller provides everything)
    mass_info: Dict[str, Any] = {}
    reaction_info: Dict[str, Any] = {}
    bolt_info: Dict[str, Any] = {}
    opt_info: Dict[str, Any] = {}

    if material_dictionary and material_free and material_bracket and free_objects:

        solid_objects = [geom.solid(free_object) for free_object in free_objects]

        log(solid_objects)
        
        free_density = float(material_dictionary[material_free]["Density"])
        bracket_density = float(material_dictionary[material_bracket]["Density"])

        free_volume = sum(obj.volume() for obj in solid_objects) / 1_000_000_000.0

        log(f"free_volume: {free_volume}")
        print(f"free_volume: {free_volume}")

        free_mass = free_volume * free_density

        bracket_volume = bracket_solid.volume() / 1_000_000_000.0
        bracket_mass = bracket_volume * bracket_density

        combined_mass = free_mass + bracket_mass
        gravity_force_1 = free_mass * 9.81
        gravity_force_2 = combined_mass * 9.81
        
        free_object = geom.solid(free_objects[0])

        free_cg = free_object.centroid()
        bracket_cg = bracket_solid.centroid()

        combined_cg = geom.point(
            (free_mass * free_cg.x + bracket_mass * bracket_cg.x) / combined_mass,
            (free_mass * free_cg.y + bracket_mass * bracket_cg.y) / combined_mass,
            (free_mass * free_cg.z + bracket_mass * bracket_cg.z) / combined_mass,
        )

        mass_info = {
            "free_mass": free_mass,
            "bracket_mass": bracket_mass,
            "combined_mass": combined_mass,
            "gravity_force_1": gravity_force_1,
            "gravity_force_2": gravity_force_2,
            "free_cg": free_cg,
            "bracket_cg": bracket_cg,
            "combined_cg": combined_cg,
        }

        log(mass_info)
        print(mass_info)

        # ------------------------------------------------------------------
        # Solver helpers
        # ------------------------------------------------------------------

        def compute_reactions(calculate_equilibrium_reactions, vector_magnitude,
                            applied_F, applied_M, reaction_pts_xyz_m, cg_xyz_m):
            reactions = calculate_equilibrium_reactions(applied_F, applied_M, reaction_pts_xyz_m, cg_xyz_m)
            log("---- Reaction Points and Their Reaction Loads ----")
            for i, (pt, r) in enumerate(zip(reaction_pts_xyz_m, reactions), start=1):
                mag = vector_magnitude(r)
                log(f"Support {i} at {pt} -> Reaction Force: {r} N, Magnitude: {mag:.2f} N")
            return reactions

        def reactions_to_T_S_vectors(split_vector, reactions_xyz, perp_vec_xyz):
            T, S = [], []
            v = perp_vec_xyz
            for reaction in reactions_xyz:
                u = reaction
                t, s = split_vector(u, v)
                T.append(t)
                S.append(s)
            return T, S

        # ------------------------------------------------------------------
        # External solvers
        # ------------------------------------------------------------------
        
        from solvers.rigid_body_solver import calculate_equilibrium_reactions, split_vector, vector_magnitude
        from solvers.bolt_sizing import select_bolt_gh
        from solvers.section_sizing import optimum_thickness
        

        applied_F = [0, 0, gravity_force_1]
        print("applied_F:", applied_F)

        applied_M = [0, 0, 0]

        _free_cg = free_cg._serialize()
        cg_free = [_free_cg[0]/1000.0, _free_cg[1]/1000.0, _free_cg[2]/1000.0]

        _cps_bolts_2 = [pt._serialize() for pt in cps_bolts_2]
        reaction_points_2 = [[pt[0]/1000.0, pt[1]/1000.0, pt[2]/1000.0] for pt in _cps_bolts_2]
        print("reaction_points_2:", reaction_points_2)
        
        R_2 = compute_reactions(calculate_equilibrium_reactions, vector_magnitude,
                                        applied_F, applied_M, reaction_points_2, cg_free)
        print("reactions_2:", R_2)

        # R_2 = [geom.vector(r[0], r[1], r[2]) for r in reactions_2]
        # print("R_2:", [v._serialize() for v in R_2])

        _perp_2 = perp_2._serialize()
        T_2, S_2 = reactions_to_T_S_vectors(split_vector, R_2, [_perp_2[0], _perp_2[1], _perp_2[2]])

        # T_2 = [geom.vector(t[0], t[1], t[2]) for t in T2_xyz]
        # print(T_2)
        # S_2 = [geom.vector(s[0], s[1], s[2]) for s in S2_xyz]
        # print(S_2)

        # Bolt sizing #

        bolt_summary, bolt_sizes, bolt_hole_sizes = [], [], []
        for T, S in zip(T_2, S_2):
            result, summary = select_bolt_gh(np.linalg.norm(T), np.linalg.norm(S))
            bolt_summary.append(summary)
            bolt_sizes.append(result.size)
            bolt_hole_sizes.append(result.hole_size)
        hole_size_2 = max(bolt_hole_sizes)

        applied_F = [0, 0, gravity_force_2]
        applied_M = [0, 0, 0]
        
        _combined_cg = combined_cg._serialize()
        cg_combined = [_combined_cg[0]/1000.0, _combined_cg[1]/1000.0, _combined_cg[2]/1000.0]

        _cps_bolts_1 = [pt._serialize() for pt in cps_bolts_2]
        reaction_points_1 = [[pt[0]/1000.0, pt[1]/1000.0, pt[2]/1000.0] for pt in _cps_bolts_1]
        R_1 = compute_reactions(calculate_equilibrium_reactions, vector_magnitude,
                                        applied_F, applied_M, reaction_points_1, cg_combined)
        
        print("reactions_1:", R_1)

        # R_1 = [geom.vector(r[0], r[1], r[2]) for r in reactions_1]
        _perp_1 = perp_1._serialize()
        T_1, S_1 = reactions_to_T_S_vectors(split_vector, R_1, [_perp_1[0], _perp_1[1], _perp_1[2]])
        # T_1 = [geom.vector(t[0], t[1], t[2]) for t in T1_xyz]
        # S_1 = [geom.vector(s[0], s[1], s[2]) for s in S1_xyz]

        print("T_1:", T_1)
        print("S_1:", S_1)

        reaction_info = {
            "R_1": R_1.tolist(),
            "T_1": [l.tolist() for l in T_1],
            "S_1": [l.tolist() for l in S_1],
            "R_2": R_2.tolist(),
            "T_2": [l.tolist() for l in T_2],
            "S_2": [l.tolist() for l in S_2],
        }

        new_bolt_hole_sizes = []
        for T, S in zip(T_1, S_1):
            result, summary = select_bolt_gh(np.linalg.norm(T), np.linalg.norm(S))
            bolt_summary.append(summary)
            bolt_sizes.append(result.size)
            new_bolt_hole_sizes.append(result.hole_size)

        hole_size_1 = max(new_bolt_hole_sizes)
        bolt_hole_sizes += new_bolt_hole_sizes

        bolt_info = {
            "bolt_summary": bolt_summary,
            "bolt_sizes": bolt_sizes,
        }

        F1 = geom.vector(*T_2[1]).length() + geom.vector(*T_2[3]).length()
        F2 = geom.vector(*T_2[0]).length() + geom.vector(*T_2[2]).length()
        L1 = cp_bolts_2_2
        L2 = cp_bolts_2_1
        M = (F1 * L1) - (F2 * L2)

        print("M:", type(M))

        bb = bracket_solid.bounding_box(plane_2)
        print(bb)
        bb_height = (bb["max"][2] - bb["min"][2])
        print("bb_height: ", bb_height)

        bracket_yield_strength = float(material_dictionary[material_bracket]["Yield Strength"])
        b = bracket_width / 1000.0
        h = bb_height / 1000.0
        s = bracket_yield_strength * 1_000_000.0

        # Optimize thickness #

        t_opt = optimum_thickness(b, h, M, s) * 1000.0

        print("t_opt:", t_opt)

        opt_info = {
            "bb": bb,
            "t_opt": t_opt,
        }
    
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
    }