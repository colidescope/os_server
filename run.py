import sys, os, json, pickle

from helpers import log
from geom.occ_helpers import mesh_surface_to_occ_face, face_area, mesh_to_occ_solid
from geom.occ_backend import OCCBackend
from scripts.bracket import main
# from scripts.test import main

def get_inputs(data_path):

    input_set = []
    item_paths = []

    if os.path.isdir(data_path):
        log(f"{data_path} is a directory.")

        # Get a list of all items (files and subdirectories) in the directory
        items = os.listdir(data_path)
        
        for item in items:
            item_path = os.path.join(data_path, item)  # Create full path to the item
            
            if os.path.isfile(item_path):  # Check if the item is a file

                item_paths.append(item_path)

                with open(item_path, 'r') as f:
                    json_data = json.load(f)

                    ## single item
                    if not isinstance(json_data, list):
                        input_set += [json_data]
                    else:
                        input_set += json_data

            else:
                log(f"{item} is a subdirectory.")

    elif os.path.isfile(data_path):
        log(f"{data_path} is a file.")

        item_paths.append(data_path)

        with open(data_path, 'r') as f:
            json_data = json.load(f)

            ## single item
            if not isinstance(json_data, list):
                input_set += [json_data]
            else:
                input_set += json_data
        
    else:
        log(f"{data_path} does not exist.")
    
    return input_set, item_paths

def get_all_data(inputs):

    # data_all = []

    # geo = inputs.get("geo")

    log(inputs)

    face_1_geo_data = inputs.get("srf_1").get("geometry")
    face_2_geo_data = inputs.get("srf_2").get("geometry")
    free_objects_data = [object_data.get("geometry") for object_data in inputs.get("free_objects")]

    # 1) build planar faces from mesh payload
    face1 = mesh_surface_to_occ_face(
        positions=face_1_geo_data.get("positions"),
        indices=face_1_geo_data.get("indices"),
        item_size=face_1_geo_data.get("itemSize"),
    )
    face2 = mesh_surface_to_occ_face(
        positions=face_2_geo_data.get("positions"),
        indices=face_2_geo_data.get("indices"),
        item_size=face_2_geo_data.get("itemSize"),
    )

    solids = [mesh_to_occ_solid(
        positions=geo.get("positions"),
        indices=geo.get("indices"),
        item_size=geo.get("itemSize"),
        tol=1e-6,
        assume_positions_are_zxy=False
    ) for geo in free_objects_data]  # set True if your vertices are [Z,X,Y]

    log(solids)

    a1 = face_area(face1)
    a2 = face_area(face2)

    log(f"face1 area: {a1:.6f}")
    log(f"face2 area: {a2:.6f}")

    geom = OCCBackend()

    result = main(
        geom=geom,
        srf_1=face1,
        srf_2=face2,
        free_objects=solids,          # if you have them later, pass shapes here

        min_edge_clearance=inputs.get("min_edge_clearance"),
        num_ribs=inputs.get("num_ribs"),
        init_plate_thickness=inputs.get("init_plate_thickness"),
        
        bolt_spacing_1=inputs.get("bolt_spacing_1"),
        bolt_spacing_2=inputs.get("bolt_spacing_2"),
        init_bolt_radius=inputs.get("init_bolt_radius"),
        max_bolt_dia=inputs.get("max_bolt_dia"),

        material_dictionary=inputs.get("material_dictionary"),
        material_free=inputs.get("material_free"),
        material_bracket=inputs.get("material_bracket"),
    )

    # log(result)
    # print(result)
    
    return result


if __name__ == '__main__':

    if os.path.exists("log.txt"):
        os.remove("log.txt")

    message = "START RUN.PY"
    log(f"\n\n{message}\n{''.join(['-']*len(message))}\n")

    script_path = os.path.realpath(__file__).split("\\")

    params = []

    # if len(sys.argv) < 3:
    #     pass
    #     # hard code param data
    # # run from outside with args
    # else:
    #     # load data path from args
    #     params_string = sys.argv[2]
    #     if len(params_string) > 0:
    #         params = [float(param) for param in params_string.split(",")]
    #         print(params)
    # print("sys.argv", sys.argv)
    if len(sys.argv) < 2:
        # hard code data path
        data_path = "\\".join(script_path[:-1] + ["data", "sample.json"])
    # run from outside with args
    else:
        # load data path from args
        data_path = sys.argv[1]

    input_set, item_paths = get_inputs(data_path)
    result = get_all_data(input_set[0])

    data = {
        "plane_1": result.get("plane_1")._serialize(),
        "plane_2": result.get("plane_2")._serialize(),
        "perp_1": result.get("perp_1")._serialize(),
        "perp_2": result.get("perp_2")._serialize(),
        "boundary_1": result.get("boundary_1")._serialize(),
        "boundary_2": result.get("boundary_2")._serialize(),
        "raw_intersection": result.get("raw_intersection")._serialize(),
        "far_edge_1": result.get("far_edge_1")._serialize(),
        "far_edge_2": result.get("far_edge_2")._serialize(),
        "intersect_line": result.get("intersect_line")._serialize(),
        "placement_edge": result.get("placement_edge")._serialize(),
        "mid_extension_1": result.get("mid_extension_1")._serialize(),
        "mid_extension_2": result.get("mid_extension_2")._serialize(),
        "axis_0": result.get("axis_0")._serialize(),
        "cps_bolts_1": [pt._serialize() for pt in result.get("cps_bolts_1")],
        "cps_bolts_2": [pt._serialize() for pt in result.get("cps_bolts_2")],
        "profile_curve": result.get("profile_curve")._serialize(),
        "inside_poly_curve": result.get("inside_poly_curve")._serialize(),
        "bracket_solid": result.get("bracket_solid")._serialize(),
    }

    mass_data = result.get("mass")
    if mass_data:
        data["mass"] = mass_data

        if "free_cg" in mass_data.keys():
            data["mass"]["free_cg"] = mass_data["free_cg"]._serialize()

        if "combined_cg" in mass_data.keys():
            data["mass"]["combined_cg"] = mass_data["combined_cg"]._serialize()

        if "bracket_cg" in mass_data.keys():
            data["mass"]["bracket_cg"] = mass_data["bracket_cg"]._serialize()

    reactions_data = result.get("reactions")
    if reactions_data:
        data["reactions"] = reactions_data

    bolts_data = result.get("bolts")
    if bolts_data:
        data["bolts"] = bolts_data

    opt_data = result.get("opt")
    if opt_data:
        data["opt"] = opt_data

        # for key in ["R_1", "T_1", "S_1", "R_2", "T_2", "S_2"]:
        #     if key in reactions_data.keys():
        #         data["reactions"][key] = [vec._serialize() for vec in reactions_data[key]]

    # log(data)

    # print(data)

    output_path = "\\".join(script_path[:-1] + ["data", "results.json"])

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print("Done with output:")

    # must be last print statement
    print(output_path)
















